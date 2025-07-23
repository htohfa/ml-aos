import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Any, Dict

from ml_aos.utils import transform_inputs


class RealDonuts(Dataset):
    """Real LSST AOS donuts from commissioning data."""

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        data_path: str = "/astro/store/epyc/ssd/users/jfc20/lsstcam_aos_data/aggregated_aos_data_april.pkl",
        **kwargs: Any,
    ) -> None:
        """Load real LSST AOS donuts.

        Parameters
        ----------
        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        transform: bool, default=True
            Whether to apply transform_inputs from ml_aos.utils.
        data_path: str
            Path to the pickle file containing the data.
        """
        # Save the settings
        self.settings = {
            "mode": mode,
            "transform": transform,
        }

        # Load all data
        self.data = pd.read_pickle(data_path)
        
        # Get unique visits for splitting
        unique_visits = self.data['visit'].unique()
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(unique_visits)
        
        # Split visits 80/10/10
        n_visits = len(unique_visits)
        train_end = int(n_visits * 0.8)
        val_end = train_end + int(n_visits * 0.1)
        
        train_visits = unique_visits[:train_end]
        val_visits = unique_visits[train_end:val_end]
        test_visits = unique_visits[val_end:]
        
        self.visit_splits = {
            "train": train_visits,
            "val": val_visits,
            "test": test_visits,
        }

        # Filter data for the current mode
        mode_visits = self.visit_splits[mode]
        self.mode_data = self.data[self.data['visit'].isin(mode_visits)].reset_index(drop=True)

    def _reconcile_zernikes(self, zk_sparse: np.ndarray) -> np.ndarray:
        """Convert sparse Zernike array to full array expected by network.
        
        Input: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28]
        Output: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        """
        # Initialize output array with NaN for indices 4-22 (19 total)
        zk_full = np.full(19, np.nan, dtype=np.float32)
        
        # Map the available values
        # Indices 0-11 map directly (Noll 4-15)
        zk_full[0:12] = zk_sparse[0:12]
        
        # Skip indices 12-15 (Noll 16-19) - leave as NaN
        
        # Indices 16-18 come from sparse indices 12-14 (Noll 20-22)
        zk_full[16:19] = zk_sparse[12:15]
        
        # Drop indices 27, 28 (sparse indices 15, 16)
        
        return zk_full

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.mode_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return donut corresponding to the index."""
        # Get the row for this index
        row = self.mode_data.iloc[idx]
        
        # Get the image
        img = np.array(row['image'])
        
        # Get field coordinates (convert degrees to radians)
        fx = float(row['field_x']) * np.pi / 180
        fy = float(row['field_y']) * np.pi / 180

        # Get the intra/extra flag
        intra = (row['defocal_type'] == 'intra')

        # Get the observed band - convert to index
        band_str = str(row['band'])
        band = "ugrizy".index(band_str)

        # Get zernikes and reconcile with network expectations
        zk_sparse = np.array(row['zk_micron'])
        zernikes = self._reconcile_zernikes(zk_sparse)

        # Standardize all the inputs for the neural net
        if self.settings["transform"]:
            img, fx, fy, intra, band = transform_inputs(
                img, fx, fy, intra, band
            )

        # Convert everything to tensors
        img = torch.from_numpy(img).float()
        fx = torch.FloatTensor([fx])
        fy = torch.FloatTensor([fy])
        intra = torch.FloatTensor([intra])
        band = torch.FloatTensor([band])
        zernikes = torch.from_numpy(zernikes).float()

        output = {
            "image": img,
            "field_x": fx,
            "field_y": fy,
            "intrafocal": intra,
            "band": band,
            "zernikes": zernikes,
        }

        return output
