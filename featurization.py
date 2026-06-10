import os
import pickle
from typing import List, Tuple, Union, Dict, Any

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem import MACCSkeys
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


# =========================================================
# RDKit pharmacophore feature factory
# =========================================================

fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


# =========================================================
# Feature dimensions
# =========================================================

ATOM_FDIM = 89
BOND_FDIM = 9

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048

# AtomPair 2048 + MACCS 167 + MorganBits 2048 + MorganCounts 2048 + Pharm 27
FP_FDIM = 2048 + 167 + 2048 + 2048 + 27


FEATURES = {
    "atomic_num": [
        1, 3, 4, 5, 6, 7, 8, 9,
        11, 12, 13, 14, 15, 16, 17,
        19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35,
        38, 39, 40, 42, 43, 46, 47, 48, 49,
        50, 51, 53, 56, 57, 60, 62, 63, 64,
        66, 70, 78, 79, 80, 81, 82, 83, 88, 98
    ],
    "degree": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-1, -2, 1, 2, 3, 0],
    "chiral_tag": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "stereo": [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}


# =========================================================
# Basic feature dimension getters
# =========================================================

def get_atom_fdim() -> int:
    """
    Return atom feature dimensionality.
    """
    return ATOM_FDIM


def get_bond_fdim(atom_messages: bool = False) -> int:
    """
    Return bond feature dimensionality.

    If atom_messages=False, bond features include atom features of the source atom.
    If atom_messages=True, only bond features are used.
    """
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()


def get_fp_fdim() -> int:
    """
    Return molecular fingerprint feature dimensionality.
    """
    return FP_FDIM


# =========================================================
# One-hot encoding
# =========================================================

def onek_encoding_unk(value: Any, choices: List[Any]) -> List[int]:
    """
    Creates a one-hot encoding with an extra unknown category.

    The output length is len(choices) + 1.
    If value is not in choices, the last element is set to 1.
    """
    encoding = [0] * (len(choices) + 1)

    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1

    return encoding


# =========================================================
# Atom and bond features
# =========================================================

def atom_features(
    atom: Chem.rdchem.Atom,
    functional_groups: List[int] = None
) -> List[Union[bool, int, float]]:
    """
    Build atom feature vector.
    """
    features = (
        onek_encoding_unk(atom.GetAtomicNum(), FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), FEATURES["formal_charge"])
        + onek_encoding_unk(atom.GetChiralTag(), FEATURES["chiral_tag"])
        + onek_encoding_unk(atom.GetTotalNumHs(), FEATURES["num_Hs"])
        + onek_encoding_unk(atom.GetHybridization(), FEATURES["hybridization"])
        + [1 if atom.GetIsAromatic() else 0]
    )

    if functional_groups is not None:
        features += functional_groups

    return [float(x) for x in features]


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Build bond feature vector.
    """
    if bond is None:
        fbond = [0.0] * BOND_FDIM
    else:
        bt = bond.GetBondType()

        fbond = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ]

        fbond += onek_encoding_unk(bond.GetStereo(), FEATURES["stereo"])

    return [float(x) for x in fbond]


# =========================================================
# Pharmacophore fingerprints
# =========================================================

def pharm_feats(mol: Chem.Mol, feature_factory=factory) -> List[int]:
    """
    Generate pharmacophore feature vector from RDKit BaseFeatures.
    Default dimension is usually 27.
    """
    try:
        feature_defs = list(feature_factory.GetFeatureDefs().keys())

        types = []
        for key in feature_defs:
            if "." in key:
                types.append(key.split(".")[1])
            else:
                types.append(key)

        feats = [feat.GetType() for feat in feature_factory.GetFeaturesForMol(mol)]

        result = [0] * len(types)
        feat_set = set(feats)

        for i, feature_type in enumerate(types):
            if feature_type in feat_set:
                result[i] = 1

        # Keep dimension fixed at 27 for compatibility
        if len(result) < 27:
            result += [0] * (27 - len(result))
        elif len(result) > 27:
            result = result[:27]

        return result

    except Exception:
        return [0] * 27


# =========================================================
# Molecular fingerprints
# =========================================================

def get_fp_feature(mol: Chem.Mol) -> List[float]:
    """
    Generate fixed molecular fingerprint features.

    Components:
    - Hashed atom-pair fingerprint: 2048
    - MACCS keys: 167
    - Morgan bit fingerprint: 2048
    - Hashed Morgan count fingerprint: 2048
    - Pharmacophore fingerprint: 27

    Total dimension: 6338
    """
    if mol is None:
        return [0.0] * FP_FDIM

    try:
        fp_atom_pairs = list(
            rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol,
                nBits=2048
            )
        )
    except Exception:
        fp_atom_pairs = [0] * 2048

    try:
        fp_maccs = list(MACCSkeys.GenMACCSKeys(mol))
    except Exception:
        fp_maccs = [0] * 167

    try:
        fp_morgan_bits = list(
            GetMorganFingerprintAsBitVect(
                mol,
                radius=MORGAN_RADIUS,
                nBits=MORGAN_NUM_BITS
            )
        )
    except Exception:
        fp_morgan_bits = [0] * 2048

    try:
        fp_morgan_counts = list(
            AllChem.GetHashedMorganFingerprint(
                mol,
                radius=MORGAN_RADIUS,
                nBits=MORGAN_NUM_BITS
            )
        )
    except Exception:
        fp_morgan_counts = [0] * 2048

    try:
        fp_pharm = pharm_feats(mol)
    except Exception:
        fp_pharm = [0] * 27

    fp = fp_atom_pairs + fp_maccs + fp_morgan_bits + fp_morgan_counts + fp_pharm

    if len(fp) < FP_FDIM:
        fp += [0] * (FP_FDIM - len(fp))
    elif len(fp) > FP_FDIM:
        fp = fp[:FP_FDIM]

    return [float(x) for x in fp]


# =========================================================
# Motif decomposition
# =========================================================

def get_cliques_link(breaks: List[List[int]], cliques: List[List[int]]) -> Dict[str, List[int]]:
    """
    Construct links between BRICS-cleaved motifs.
    """
    clique_id_of_node = {}

    for idx, clique in enumerate(cliques):
        for atom_idx in clique:
            clique_id_of_node[atom_idx] = idx

    breaks_bond = {}

    for bond in breaks:
        a1, a2 = bond[0], bond[1]

        if a1 in clique_id_of_node and a2 in clique_id_of_node:
            breaks_bond[f"{a1}_{a2}"] = [
                clique_id_of_node[a1],
                clique_id_of_node[a2],
            ]

    return breaks_bond


def motif_decomp(mol: Chem.Mol) -> Tuple[Dict[str, List[int]], List[List[int]]]:
    """
    BRICS-based motif decomposition.

    Returns:
    - breaks: dictionary describing motif links
    - cliques: list of atom index groups
    """
    if mol is None:
        return {}, []

    n_atoms = mol.GetNumAtoms()

    if n_atoms <= 1:
        return {}, []

    cliques = []
    breaks = []

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    try:
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
    except Exception:
        brics_bonds = []

    if len(brics_bonds) != 0:
        for bond in brics_bonds:
            a1, a2 = bond[0][0], bond[0][1]

            if [a1, a2] in cliques:
                cliques.remove([a1, a2])
            elif [a2, a1] in cliques:
                cliques.remove([a2, a1])

            cliques.append([a1])
            cliques.append([a2])
            breaks.append([a1, a2])

    # Merge overlapping cliques
    changed = True
    while changed:
        changed = False
        new_cliques = []

        while cliques:
            current = set(cliques.pop(0))
            merged = True

            while merged:
                merged = False
                remaining = []

                for clique in cliques:
                    clique_set = set(clique)

                    if len(current & clique_set) > 0:
                        current |= clique_set
                        merged = True
                        changed = True
                    else:
                        remaining.append(clique)

                cliques = remaining

            new_cliques.append(list(current))

        cliques = new_cliques

    cliques = [c for c in cliques if 0 < len(c) < n_atoms]

    breaks_bond = get_cliques_link(breaks, cliques)

    return breaks_bond, cliques


# =========================================================
# MolGraph object
# =========================================================

class MolGraph:
    """
    Molecular graph for a single molecule.
    Compatible with FH-GNN / FEHGNN BatchMolGraph.
    """

    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)

        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        self.n_atoms = self.mol.GetNumAtoms()
        self.n_bonds = 0

        self.f_atoms = []
        self.f_bonds = []
        self.a2b = [[] for _ in range(self.n_atoms)]
        self.b2a = []
        self.b2revb = []

        # Atom features
        for atom in self.mol.GetAtoms():
            self.f_atoms.append(atom_features(atom))

        # Directed bond features
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()

            f_bond = bond_features(bond)

            # a1 -> a2
            self.f_bonds.append(self.f_atoms[a1] + f_bond)
            self.b2a.append(a1)
            self.a2b[a2].append(self.n_bonds)
            b1 = self.n_bonds
            self.n_bonds += 1

            # a2 -> a1
            self.f_bonds.append(self.f_atoms[a2] + f_bond)
            self.b2a.append(a2)
            self.a2b[a1].append(self.n_bonds)
            b2 = self.n_bonds
            self.n_bonds += 1

            self.b2revb.append(b2)
            self.b2revb.append(b1)

        self.fp_x = torch.FloatTensor([get_fp_feature(self.mol)])

        # Keep compatibility with original FH-GNN code
        self.num_part = torch.LongTensor([[self.n_atoms]])


# =========================================================
# Batch molecular graph
# =========================================================

class BatchMolGraph:
    """
    Batch of molecular graphs.

    This class first tries to load preprocessed molecular graphs from:
        dataset/{data_name}/raw/process_all.pkl

    If the file does not exist, it dynamically builds molecular graphs from SMILES.
    """

    def __init__(
        self,
        smiles,
        atom_fdim: int = ATOM_FDIM,
        bond_fdim: int = ATOM_FDIM + BOND_FDIM,
        fp_fdim: int = FP_FDIM,
        data_name: str = "demo",
        data_dir: str = "./dataset"
    ):
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.fp_fdim = fp_fdim

        # Start with zero padding
        self.n_atoms = 1
        self.n_bonds = 1

        self.a_scope = []
        self.b_scope = []

        f_atoms = [[0.0] * self.atom_fdim]
        f_bonds = [[0.0] * self.bond_fdim]

        a2b = [[]]
        b2a = [0]
        b2revb = [0]

        fp_x_out = torch.empty((0, self.fp_fdim))

        # Load preprocessed graph cache if available
        cache_path = os.path.join(data_dir, data_name, "raw", "process_all.pkl")

        mol_graphs = {}

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    mol_graphs = pickle.load(f)
            except Exception:
                mol_graphs = {}

        mol_atom_num = []

        for smi in smiles:
            # Compatible with original FH-GNN where smiles may be tuple/list
            if isinstance(smi, (list, tuple)):
                smi_key = smi[0]
            else:
                smi_key = smi

            if smi_key in mol_graphs:
                mol_graph = mol_graphs[smi_key]
            else:
                mol_graph = MolGraph(smi_key)

            mol_atom_num.append(int(mol_graph.n_atoms))

            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))

            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

            if hasattr(mol_graph, "fp_x"):
                fp_x = mol_graph.fp_x
                if fp_x.dim() == 1:
                    fp_x = fp_x.unsqueeze(0)
            else:
                mol = Chem.MolFromSmiles(smi_key)
                fp_x = torch.FloatTensor([get_fp_feature(mol)])

            if fp_x.size(1) < self.fp_fdim:
                pad = torch.zeros(fp_x.size(0), self.fp_fdim - fp_x.size(1))
                fp_x = torch.cat([fp_x, pad], dim=1)
            elif fp_x.size(1) > self.fp_fdim:
                fp_x = fp_x[:, :self.fp_fdim]

            fp_x_out = torch.cat((fp_x_out, fp_x), dim=0)

        self.max_num_bonds = max(
            1,
            max(len(in_bonds) for in_bonds in a2b)
        )

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)

        self.a2b = torch.LongTensor([
            a2b[a] + [0] * (self.max_num_bonds - len(a2b[a]))
            for a in range(self.n_atoms)
        ])

        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)

        self.b2b = None
        self.a2a = None

        self.smiles = smiles
        self.mol_atom_num = mol_atom_num
        self.fp_x = fp_x_out

    def get_components(
        self,
        atom_messages: bool = False
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        List[Tuple[int, int]],
        List[Tuple[int, int]]
    ]:
        """
        Return graph tensors.
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, :get_bond_fdim(atom_messages=True)]
        else:
            f_bonds = self.f_bonds

        return (
            self.f_atoms,
            f_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
        )

    def get_b2b(self) -> torch.LongTensor:
        """
        Return bond-to-bond mapping.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]

            revmask = (
                b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))
            ).long()

            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Return atom-to-atom mapping.
        """
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]

        return self.a2a


# =========================================================
# Optional preprocessing helper
# =========================================================

def build_process_all(
    smiles_list: List[str],
    data_name: str = "demo",
    data_dir: str = "./dataset"
) -> Dict[str, MolGraph]:
    """
    Build and save process_all.pkl for faster future loading.

    Save path:
        dataset/{data_name}/raw/process_all.pkl
    """
    save_dir = os.path.join(data_dir, data_name, "raw")
    os.makedirs(save_dir, exist_ok=True)

    mol_graphs = {}

    for smi in smiles_list:
        try:
            mol_graphs[smi] = MolGraph(smi)
        except Exception as e:
            print(f"[Warning] Failed to process SMILES {smi}: {e}")

    save_path = os.path.join(save_dir, "process_all.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(mol_graphs, f)

    print(f"Saved processed molecular graphs to: {save_path}")

    return mol_graphs