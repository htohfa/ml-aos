import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
from glob import glob

from google.colab import drive
drive.mount('/content/drive')
input_dir = '/content/drive/My Drive/rubin_aos_batoid_sims 2'

# Your existing load_data function
def load_data(path: str = '/content/drive/My Drive/rubin_aos_batoid_sims 2', chunks: int = 100):
    """Load the table of batoid simulations.
    
    Parameters
    ----------
    path : str, default="data"
        Path to where the data is stored.
    chunks : int, default=100
        Number of chunks to load, where each chunk corresponds to a different
        atmosphere. Must be between 1 and 100, inclusive.

    Returns
    -------
    pd.DataFrame
        Table of simulated data.
    """
    # get the list of all files
    files = glob(f"{path}/obs*.pkl")

    # check that we found the files
    if len(files) == 0:
        raise RuntimeError(
            f"Did not find data files at path '{path}'."
        )

    # select the chunks
    files = files[:chunks]

    # load and return
    return pd.concat([pd.read_pickle(file) for file in files])


class DonutsDataFrame(Dataset):
    """AOS donuts dataset from pandas DataFrame format."""

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        data_path: str = "./data",
        chunks: int = 100,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Load the simulated AOS donuts from pandas DataFrame.

        Parameters
        ----------
        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        transform: bool, default=True
            Whether to apply transforms to inputs.
        data_path: str, default="./data"
            Path to the data directory containing obs*.pkl files.
        chunks: int, default=100
            Number of chunks to load (1-100).
        train_split: float, default=0.7
            Fraction of data for training.
        val_split: float, default=0.15
            Fraction of data for validation.
        test_split: float, default=0.15
            Fraction of data for testing.
        random_seed: int, default=42
            Random seed for reproducible splits.
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Split fractions must sum to 1.0"
        
        # Save settings
        self.settings = {
            "mode": mode,
            "transform": transform,
            "data_path": data_path,
            "chunks": chunks,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "random_seed": random_seed,
        }

        # Load the full dataset
        self.full_data = load_data(data_path, chunks)
        
        # Create train/val/test splits
        np.random.seed(random_seed)
        n_samples = len(self.full_data)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_split * n_samples)
        val_end = train_end + int(val_split * n_samples)
        
        self.splits = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:]
        }
        
        # Get the subset for this mode
        self.data = self.full_data.iloc[self.splits[mode]].reset_index(drop=True)

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return simulation corresponding to the index.

        Parameters
        ----------
        idx: int
            The index of the simulation to return.

        Returns
        -------
        dict
            The dictionary contains the following pytorch tensors based on 
            available columns in your DataFrame.
        """
        # Get the row
        row = self.data.iloc[idx]
        
        # Extract basic identifiers
        pnt_id = row['pntId']
        obs_id = row['obsId'] 
        obj_id = row['objId']
        
        # Extract field positions
        fx = row['fx']
        fy = row['fy']
        
        # Extract coordinates (ra, dec)
        ra = row['ra']
        dec = row['dec']
        
        # Extract corner and intrafocal information
        corner = row['corner']  # This appears to be detector corner (e.g., 'R00')
        intra = row['intra']    # Boolean for intrafocal/extrafocal
        
        # Apply transforms if requested
        if self.settings["transform"]:
            # You would implement your transform_inputs function here
            # For now, just normalize field positions as an example
            fx = fx / 2.0  # Normalize to roughly [-1, 1] range
            fy = fy / 2.0
            intra = float(intra)  # Convert boolean to float
        
        # Convert to tensors
        output = {
            "field_x": torch.FloatTensor([fx]),
            "field_y": torch.FloatTensor([fy]),
            "ra": torch.FloatTensor([ra]),
            "dec": torch.FloatTensor([dec]),
            "intrafocal": torch.FloatTensor([intra]),
            "corner": corner,  # Keep as string or convert to categorical
            "pntId": torch.LongTensor([pnt_id]),
            "obsId": torch.LongTensor([obs_id]),
            "objId": torch.LongTensor([obj_id]),
        }
        
        # Add any additional columns that might be in your DataFrame
        for col in self.data.columns:
            if col not in ['pntId', 'obsId', 'objId', 'fx', 'fy', 'ra', 'dec', 'corner', 'intra']:
                value = row[col]
                if isinstance(value, (int, float, np.number)):
                    output[col] = torch.FloatTensor([value])
                # Add other type conversions as needed
        
        return output


if __name__ == "__main__":
    # Create datasets
    train_dataset = DonutsDataFrame(mode="train", data_path="./data", chunks=10)
    val_dataset = DonutsDataFrame(mode="val", data_path="./data", chunks=10)
    test_dataset = DonutsDataFrame(mode="test", data_path="./data", chunks=10)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get a sample
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Field position: ({sample['field_x'].item():.3f}, {sample['field_y'].item():.3f})")
    print(f"Intrafocal: {sample['intrafocal'].item()}")
