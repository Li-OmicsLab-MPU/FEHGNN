# FH-GNN

## Introduction
* Source code for the paper "Fingerprint-enhanced hierarchical molecular graph neural networks for property prediction".

## Environment
* base dependencies:
```
  - numpy == 1.21.5
  - rdkit == 2018.03.4
  - pandas == 1.3.5
  - python == 3.7.16
  - pytorch == 1.12.1
  - scikit-learn == 1.0.2
```

## Usage

#### Args:
- --dataset : The name of input dataset.
- --data_dir : The path of input CSV file.
- --save_dir : The path to save output model.
- --batch_size : The input batch size for training.
- --epochs : The number of epochs to train.
- --lr : The learning rate for the prediction layer.
- --depth : The depth of molecule encoder.

#### Quick Run
```bash
python train.py
```
