import numpy as np
import torch
from torch.utils.data import Subset
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles, include_chirality=True):
    """
    Generate Bemis-Murcko scaffold from a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return ""

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=include_chirality
    )

    return scaffold


def random_split(
    dataset,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=0
):
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    n_total = len(dataset)
    indices = np.arange(n_total)

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_train = int(np.floor(frac_train * n_total))
    n_valid = int(np.floor(frac_valid * n_total))

    train_idx = indices[:n_train].tolist()
    valid_idx = indices[n_train:n_train + n_valid].tolist()
    test_idx = indices[n_train + n_valid:].tolist()

    return (
        Subset(dataset, train_idx),
        Subset(dataset, valid_idx),
        Subset(dataset, test_idx),
    )


def scaffold_split(
    dataset,
    smiles_list,
    task_idx=None,
    null_value=0,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=0,
    include_chirality=True,
    fallback_to_random=True
):
    """
    Split a dataset by Bemis-Murcko scaffolds.

    Args:
        dataset: Dataset object.
        smiles_list: List of SMILES strings corresponding to dataset samples.
        task_idx: Optional task index for multi-task labels.
        null_value: Label value treated as missing when task_idx is provided.
        frac_train: Fraction of training samples.
        frac_valid: Fraction of validation samples.
        frac_test: Fraction of test samples.
        seed: Random seed.
        include_chirality: Whether to include chirality in scaffold generation.
        fallback_to_random: Use random split if scaffold split produces empty valid/test sets.

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if len(dataset) != len(smiles_list):
        raise ValueError(
            f"Dataset size and SMILES list size do not match: "
            f"{len(dataset)} vs {len(smiles_list)}"
        )

    if len(dataset) < 3:
        raise ValueError("Dataset must contain at least 3 samples for splitting.")

    valid_indices = []

    if task_idx is not None:
        for idx in range(len(dataset)):
            label = dataset[idx][1]

            if torch.is_tensor(label):
                label_value = label[task_idx].item()
            elif isinstance(label, (list, tuple, np.ndarray)):
                label_value = label[task_idx]
            else:
                label_value = label

            if label_value != null_value:
                valid_indices.append(idx)
    else:
        valid_indices = list(range(len(dataset)))

    scaffolds = defaultdict(list)

    for idx in valid_indices:
        smiles = smiles_list[idx]
        scaffold = generate_scaffold(
            smiles,
            include_chirality=include_chirality
        )
        scaffolds[scaffold].append(idx)

    scaffold_sets = list(scaffolds.values())

    scaffold_sets = sorted(
        scaffold_sets,
        key=lambda x: (len(x), x[0]),
        reverse=True
    )

    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_sets)

    n_total = len(valid_indices)
    n_valid = int(np.floor(frac_valid * n_total))
    n_test = int(np.floor(frac_test * n_total))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    if fallback_to_random and (len(valid_idx) == 0 or len(test_idx) == 0):
        return random_split(
            dataset=dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=seed
        )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, valid_idx),
        Subset(dataset, test_idx),
    )