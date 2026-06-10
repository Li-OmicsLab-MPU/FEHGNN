## Introduction
Source code for the demo implementation of **Fingerprint-Enhanced Hyperbolic Graph Neural Network (FEHGNN)**.

FEHGNN is designed for molecular property prediction by integrating molecular graph representations, molecular fingerprints, and hyperbolic representation learning. The model first encodes molecular structures through graph neural message passing, then projects graph-level embeddings into a Poincare-ball-inspired hyperbolic representation space, and finally fuses the hyperbolic graph embedding with molecular fingerprint features for downstream prediction.

## Model Overview

FEHGNN uses the following architecture:

```text
SMILES
  ├─ Molecular graph encoder
  │    └─ Euclidean graph embedding
  │         └─ Hyperbolic projection / tangent-space readout
  └─ Molecular fingerprint encoder
       └─ Fingerprint embedding

Hyperbolic graph embedding + fingerprint embedding
  → weighted fusion
  → prediction head
```

The implemented demo corresponds to **FEHGNN Scheme A**, which introduces a lightweight hyperbolic projection module after the graph encoder while preserving stable graph message passing.

## Main Files

```text
FEHGNN/
├── fehgnn.py                 # FEHGNN model with hyperbolic projection
├── train_fehgnn_demo.py      # Demo training script
├── loader.py                 # Dataset loader
├── featurization.py          # Atom, bond, and fingerprint feature generation
├── splitters.py              # Dataset splitting utilities
├── dataset/
│   └── demo/
│       └── raw/
│           └── demo.csv      # Demo SMILES-label dataset
└── README.md
```

## Dataset

The demo dataset should be provided as a CSV file with two columns:

```csv
smiles,label
CCO,0
CCOC(=O)c1ccccc1C(=O)O,1
```

Labels can be encoded as either `0/1` or `-1/1`. For binary classification, `0` is converted to `-1` during loading.

To use your own dataset, replace:

```text
dataset/demo/raw/demo.csv
```

with your local data file using the same column format.

## Environment

Recommended dependencies:

```text
python >= 3.7
numpy
pandas
rdkit
torch
torch-geometric
scikit-learn
tqdm
```

Install dependencies with:

```bash
pip install torch torch-geometric rdkit-pypi scikit-learn pandas tqdm numpy
```

## Usage

### Quick Run

```bash
python train_fehgnn_demo.py --dataset demo --data_dir ./dataset --epochs 5 --batch_size 8 --hidden_size 128 --depth 3
```

### Arguments

```text
--dataset      The name of the input dataset.
--data_dir     The root directory of the dataset.
--epochs       The number of training epochs.
--batch_size   The input batch size for training.
--hidden_size  The hidden dimension of the molecular graph encoder.
--depth        The depth of the molecular graph encoder.
--lr           The learning rate.
```

## Preprocessing Output

The first run will preprocess the input CSV file and generate:

```text
dataset/demo/raw/process_all.pkl
dataset/demo/processed/geometric_data_processed.pt
dataset/demo/processed/smiles.csv
```

If you replace the dataset, remove the old processed files before rerunning:

```bash
rm -rf dataset/demo/processed dataset/demo/raw/process_all.pkl
python train_fehgnn_demo.py --dataset demo --data_dir ./dataset
```

## Model Concept

FEHGNN combines three types of molecular information:

```text
1. Local atom-bond interactions from molecular graph message passing
2. Global molecular descriptors from fixed molecular fingerprints
3. Hierarchical structure-aware representations from hyperbolic projection
```

The hyperbolic projection module is applied after graph-level embedding generation, allowing the model to enhance hierarchical molecular representation while maintaining stable Euclidean message passing.

## Notes

This demo provides a minimal and stable implementation of FEHGNN Scheme A. It is intended for binary molecular property prediction and can be extended to multi-task classification or regression by modifying the output dimension and loss function.