import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init

from featurization import BatchMolGraph


def index_select_ND(source: Tensor, index: Tensor) -> Tensor:
    """
    Selects entries from source according to a multi-dimensional index tensor.

    Args:
        source: Tensor with shape [num_items, hidden_dim].
        index: LongTensor with shape [num_targets, max_degree].

    Returns:
        Tensor with shape [num_targets, max_degree, hidden_dim].
    """
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim

    target = source.index_select(dim=0, index=index.reshape(-1))
    target = target.reshape(final_size)

    return target


class HGNNEncoder(nn.Module):
    """
    FH-GNN molecular graph encoder.

    This module keeps the original directed-bond message passing logic and
    produces one Euclidean graph-level embedding for each molecule.
    """

    def __init__(
        self,
        atom_fdim: int,
        bond_fdim: int,
        hidden_size: int = 256,
        depth: int = 5,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.device = device

        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size),
            requires_grad=False,
        )

        self.W_i = nn.Linear(self.bond_fdim, self.hidden_size, bias=False)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        self.W_a = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_b = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = nn.ReLU()

    def forward(self, mol_graph: BatchMolGraph) -> Tensor:
        """
        Args:
            mol_graph: BatchMolGraph object.

        Returns:
            Molecular graph embeddings with shape [batch_size, hidden_size].
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, _ = mol_graph.get_components()

        f_atoms = f_atoms.to(self.device)
        f_bonds = f_bonds.to(self.device)
        a2b = a2b.to(self.device)
        b2a = b2a.to(self.device)
        b2revb = b2revb.to(self.device)

        if f_atoms.size(1) != self.atom_fdim:
            raise ValueError(
                f"Atom feature dimension mismatch: model expects {self.atom_fdim}, "
                f"but BatchMolGraph provides {f_atoms.size(1)}."
            )

        if f_bonds.size(1) != self.bond_fdim:
            raise ValueError(
                f"Bond feature dimension mismatch: model expects {self.bond_fdim}, "
                f"but BatchMolGraph provides {f_bonds.size(1)}."
            )

        inputs = self.W_i(f_bonds)
        message = self.act_func(inputs)

        for _ in range(self.depth - 1):
            nei_a_message = index_select_ND(message, a2b)
            a_message = nei_a_message.sum(dim=1)

            rev_message = message[b2revb]
            message = a_message[b2a] - rev_message

            message = self.W_h(message)
            message = self.act_func(inputs + message)
            message = self.dropout_layer(message)

        nei_a_message = index_select_ND(message, a2b)
        a_message = nei_a_message.sum(dim=1)

        a_input = torch.cat([f_atoms, a_message], dim=1)

        atom_hiddens = self.W_o(a_input)
        atom_hiddens = self.act_func(atom_hiddens)
        atom_hiddens = self.dropout_layer(atom_hiddens)

        mol_vecs = []

        for a_start, a_size in a_scope:
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
                continue

            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)

            att_w = torch.matmul(self.W_a(cur_hiddens), cur_hiddens.t())
            att_w = F.softmax(att_w, dim=1)

            att_hiddens = torch.matmul(att_w, cur_hiddens)
            att_hiddens = self.W_b(att_hiddens)
            att_hiddens = self.act_func(att_hiddens)
            att_hiddens = self.dropout_layer(att_hiddens)

            mol_vec = cur_hiddens + att_hiddens
            mol_vec = mol_vec.sum(dim=0) / a_size

            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs


class HyperbolicProjection(nn.Module):
    """
    Stable Poincare-ball projection block.

    Scheme A:
        Euclidean graph embedding
            -> expmap0
            -> logmap0
            -> linear transform in tangent space
            -> expmap0
            -> logmap0
            -> residual normalized output

    This block does not require geoopt or external manifold packages.
    """

    def __init__(
        self,
        dim: int,
        curvature: float = 1.0,
        eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()

        if curvature <= 0:
            raise ValueError("curvature must be positive.")

        self.dim = dim
        self.curvature = float(curvature)
        self.eps = eps

        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def project(self, x: Tensor) -> Tensor:
        """
        Projects points into the open Poincare ball.
        """
        sqrt_c = math.sqrt(self.curvature)
        max_norm = (1.0 - self.eps) / sqrt_c

        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(max_norm / x_norm, max=1.0)

        return x * scale

    def expmap0(self, u: Tensor) -> Tensor:
        """
        Exponential map at the origin from tangent space to Poincare ball.
        """
        sqrt_c = math.sqrt(self.curvature)

        u_norm = torch.norm(u, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        scaled_norm = sqrt_c * u_norm

        gamma = torch.tanh(scaled_norm) * u / scaled_norm

        return self.project(gamma)

    def logmap0(self, x: Tensor) -> Tensor:
        """
        Logarithmic map at the origin from Poincare ball to tangent space.
        """
        sqrt_c = math.sqrt(self.curvature)

        x = self.project(x)
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(self.eps)

        scaled_norm = torch.clamp(sqrt_c * x_norm, max=1.0 - self.eps)
        factor = torch.atanh(scaled_norm) / scaled_norm

        return factor * x

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: Euclidean graph embeddings with shape [batch_size, hidden_dim].

        Returns:
            Hyperbolic-enhanced tangent-space embeddings with shape
            [batch_size, hidden_dim].
        """
        h_ball = self.expmap0(h)
        h_tangent = self.logmap0(h_ball)

        h_tangent = self.linear(h_tangent)
        h_tangent = self.dropout(h_tangent)

        h_ball = self.expmap0(h_tangent)
        h_out = self.logmap0(h_ball)

        return self.norm(h + h_out)


class WeightFusion(nn.Module):
    """
    Learnable weighted fusion for multiple feature views.

    Input shape:
        [num_views, batch_size, feature_dim]

    Output shape:
        [batch_size, feature_dim]
    """

    def __init__(
        self,
        feat_views: int,
        feat_dim: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.feat_views = feat_views
        self.feat_dim = feat_dim

        self.weight = Parameter(torch.empty((feat_views,), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(feat_dim, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.weight, 1.0 / self.feat_views)

        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise ValueError(
                f"WeightFusion expects a 3D tensor [views, batch, dim], "
                f"but got shape {tuple(inputs.shape)}."
            )

        if inputs.size(0) != self.feat_views:
            raise ValueError(
                f"Expected {self.feat_views} feature views, "
                f"but got {inputs.size(0)}."
            )

        weights = F.softmax(self.weight, dim=0).view(self.feat_views, 1, 1)
        output = torch.sum(inputs * weights, dim=0)

        if self.bias is not None:
            output = output + self.bias

        return output


class MLPHead(nn.Module):
    """
    Simple prediction head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FEHGNN(nn.Module):
    """
    Fingerprint-Enhanced Hyperbolic Graph Neural Network.

    Scheme A:
        FH-GNN molecular graph encoder
            -> graph-level Euclidean embedding
            -> hyperbolic projection block
            -> fingerprint encoder
            -> learnable fusion
            -> prediction head
    """

    def __init__(
        self,
        data_name: str,
        atom_fdim: int = 95,
        bond_fdim: int = 105,
        fp_fdim: int = 6338,
        hidden_size: int = 256,
        depth: int = 5,
        out_dim: int = 2,
        dropout: float = 0.1,
        curvature: float = 1.0,
        device: str = "cpu",
        data_dir: str = "./dataset",
    ):
        super().__init__()

        self.data_name = data_name
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.fp_fdim = fp_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.out_dim = out_dim
        self.dropout = dropout
        self.curvature = curvature
        self.device = device
        self.data_dir = data_dir

        self.encoder = HGNNEncoder(
            atom_fdim=self.atom_fdim,
            bond_fdim=self.bond_fdim,
            hidden_size=self.hidden_size,
            depth=self.depth,
            dropout=self.dropout,
            device=self.device,
        )

        self.hyperbolic_projection = HyperbolicProjection(
            dim=self.hidden_size,
            curvature=self.curvature,
            eps=1e-5,
            dropout=self.dropout,
        )

        self.fp_encoder = nn.Sequential(
            nn.Linear(self.fp_fdim, 2048),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.hidden_size),
            nn.ReLU(),
        )

        self.feature_fusion = WeightFusion(
            feat_views=2,
            feat_dim=self.hidden_size,
            device=self.device,
        )

        self.prediction_head = MLPHead(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            out_dim=self.out_dim,
            dropout=self.dropout,
        )

    def _extract_smiles(self, batch) -> List[str]:
        if hasattr(batch, "smi"):
            smiles = batch.smi
        elif hasattr(batch, "smiles"):
            smiles = batch.smiles
        else:
            raise AttributeError("The input batch must contain either 'smi' or 'smiles'.")

        return smiles

    def forward(self, batch) -> Tensor:
        smiles = self._extract_smiles(batch)

        mol_batch = BatchMolGraph(
            smiles=smiles,
            atom_fdim=self.atom_fdim,
            bond_fdim=self.bond_fdim,
            fp_fdim=self.fp_fdim,
            data_name=self.data_name,
            data_dir=self.data_dir,
        )

        graph_x = self.encoder(mol_batch)
        graph_x = self.hyperbolic_projection(graph_x)

        fp_x = mol_batch.fp_x.to(self.device).to(torch.float32)

        if fp_x.size(1) != self.fp_fdim:
            raise ValueError(
                f"Fingerprint dimension mismatch: model expects {self.fp_fdim}, "
                f"but BatchMolGraph provides {fp_x.size(1)}."
            )

        fp_x = self.fp_encoder(fp_x)

        fused_x = self.feature_fusion(
            torch.stack([graph_x, fp_x], dim=0)
        )

        output = self.prediction_head(fused_x)

        return output