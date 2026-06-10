import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from fehgnn import FEHGNN
from loader import MoleculeDataset
from splitters import scaffold_split

criterion = torch.nn.BCEWithLogitsLoss(reduction='none')


def train_epoch(model, device, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='train'):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        is_valid = y ** 2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))
        loss = torch.sum(loss_mat) / torch.clamp(torch.sum(is_valid), min=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1)


def evaluate(model, device, loader):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval'):
            batch = batch.to(device)
            pred = model(batch)
            y_true.append(batch.y.view(pred.shape))
            y_score.append(pred)
    y_true = torch.cat(y_true, dim=0).cpu().numpy().reshape(-1)
    y_score = torch.cat(y_score, dim=0).cpu().numpy().reshape(-1)
    valid = y_true ** 2 > 0
    y_bin = ((y_true[valid] + 1) / 2).astype(int)
    score = y_score[valid]
    pred_bin = (1 / (1 + np.exp(-score)) >= 0.5).astype(int)

    auc = roc_auc_score(y_bin, score) if len(np.unique(y_bin)) == 2 else float('nan')
    acc = accuracy_score(y_bin, pred_bin)
    f1 = f1_score(y_bin, pred_bin, zero_division=0)
    return auc, acc, f1


def main():
    parser = argparse.ArgumentParser(description='FEHGNN demo training script')
    parser.add_argument('--dataset', type=str, default='demo')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset_root = os.path.join(args.data_dir, args.dataset)
    dataset = MoleculeDataset(dataset_root, dataset=args.dataset)
    smiles_path = os.path.join(dataset_root, 'processed', 'smiles.csv')
    smiles_list = pd.read_csv(smiles_path, header=None)[0].tolist()

    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = FEHGNN(
        data_name=args.dataset,
        atom_fdim=89,
        bond_fdim=98,
        fp_fdim=6338,
        hidden_size=args.hidden_size,
        depth=args.depth,
        device=device,
        out_dim=1,
        data_root=args.data_dir,
        curvature=1.0,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, device, train_loader, optimizer)
        val_auc, val_acc, val_f1 = evaluate(model, device, val_loader)
        test_auc, test_acc, test_f1 = evaluate(model, device, test_loader)
        print(
            f'Epoch {epoch:03d} | loss={loss:.4f} | '
            f'val_auc={val_auc:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | '
            f'test_auc={test_auc:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}'
        )


if __name__ == '__main__':
    main()
