import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
from featurization import BatchMolGraph, ATOM_FDIM, BOND_FDIM, FP_FDIM

class MoleculeLoader(Dataset):
    """
    Simplified loader for FEHGNN demo (scheme A).
    Only supports CSV with columns: smiles,label
    """
    def __init__(self, csv_path, data_name="demo", data_dir="./dataset"):
        """
        :param csv_path: path to CSV file with 'smiles' and 'label' columns
        :param data_name: dataset name (used for caching)
        :param data_dir: root dataset directory
        """
        self.csv_path = csv_path
        self.data_name = data_name
        self.data_dir = data_dir

        df = pd.read_csv(csv_path)
        if 'smiles' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain columns: smiles,label")
        self.smiles_list = df['smiles'].astype(str).tolist()
        self.labels = df['label'].replace(0, -1).tolist()  # convert 0->-1

        # check cache
        self.processed_path = os.path.join(data_dir, data_name, "raw", "process_all.pkl")
        if not os.path.exists(self.processed_path):
            os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
            self._build_cache()

        # load BatchMolGraph from pickle
        with open(self.processed_path, "rb") as f:
            self.mol_batch = pickle.load(f)

    def _build_cache(self):
        """
        Build BatchMolGraph from SMILES and save to process_all.pkl
        """
        print("Building molecular graph cache...")
        mol_batch = BatchMolGraph(
            smiles=self.smiles_list,
            atom_fdim=ATOM_FDIM,
            bond_fdim=BOND_FDIM,
            fp_fdim=FP_FDIM,
            data_name=self.data_name,
            data_dir=self.data_dir
        )
        with open(self.processed_path, "wb") as f:
            pickle.dump(mol_batch, f)
        print(f"Saved cache to {self.processed_path}")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        Return a tuple:
            BatchMolGraph object
            label (int)
        """
        return self.mol_batch, self.labels[idx]