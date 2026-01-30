import numpy as np
if not hasattr(np, '_core'):
    import numpy.core as _core
    np._core = _core
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Any, Dict
from glob import glob
import os

from ml_aos.utils import transform_inputs


class RealDonuts(Dataset):
    """Real LSST AOS donuts from commissioning data."""

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        data_dir: str = "/astro/store/epyc/ssd/users/htohfa/aos_chunks",
        filter_low_snr: bool = False,
        snr_percentile: float = 20.0,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Load real LSST AOS donuts."""
        # Save the settings
        self.settings = {
            "mode": mode,
            "transform": transform,
            "filter_low_snr": filter_low_snr,
            "snr_percentile": snr_percentile,
            "random_seed": random_seed,
        }

        # Load all chunk files
        print(f"Loading data from {data_dir}")
        chunk_files = sorted(glob(os.path.join(data_dir, "aos_chunk_*.pkl")))
        print(f"Found {len(chunk_files)} chunk files")
        
        if len(chunk_files) == 0:
            raise ValueError(f"No chunk files found in {data_dir}")
        
        all_chunks = []
        for i, chunk_file in enumerate(chunk_files):
            if i % 10 == 0:
                print(f"  Loading chunk {i+1}/{len(chunk_files)}")
            chunk = pd.read_pickle(chunk_file)
            all_chunks.append(chunk)
        
        self.data = pd.concat(all_chunks, ignore_index=True)
        print(f"Loaded {len(self.data)} total samples")
        
        # Check for donut_blur column
        if 'donut_blur' in self.data.columns:
            print(f"\nDonut blur statistics:")
            blur_valid = ~self.data['donut_blur'].isna()
            print(f"  Valid samples: {blur_valid.sum()}")
            if blur_valid.sum() > 0:
                print(f"  Min: {self.data.loc[blur_valid, 'donut_blur'].min():.3f}")
                print(f"  Max: {self.data.loc[blur_valid, 'donut_blur'].max():.3f}")
                print(f"  Mean: {self.data.loc[blur_valid, 'donut_blur'].mean():.3f}")
        else:
            print("\nWARNING: donut_blur column not found!")
        
        # Filter out worst SNR data if requested
        if filter_low_snr:
            if not self.data['snr'].isna().all():
                snr_threshold = np.percentile(self.data['snr'].dropna(), snr_percentile)
                initial_count = len(self.data)
                self.data = self.data[self.data['snr'] >= snr_threshold]
                final_count = len(self.data)
                print(f"\nSNR filtering: Removed {initial_count - final_count} samples")
            else:
                print("\nWarning: SNR data not available, skipping SNR filtering")
        
        # Split by VISIT ID to prevent data leakage
        unique_visits = self.data['visit'].unique()
        print(f"\nTotal unique visits: {len(unique_visits)}")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
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
        
        print(f"\nVisit splits:")
        print(f"  Train: {len(train_visits)} visits")
        print(f"  Val:   {len(val_visits)} visits")
        print(f"  Test:  {len(test_visits)} visits")

        # Filter data for the current mode
        mode_visits = self.visit_splits[mode]
        self.mode_data = self.data[self.data['visit'].isin(mode_visits)].reset_index(drop=True)
        
        print(f"\n{mode.upper()} dataset statistics:")
        print(f"  Total samples: {len(self.mode_data)}")
        print(f"  Unique visits: {self.mode_data['visit'].nunique()}")
        print(f"  Unique detectors: {self.mode_data['detector'].nunique()}")
        print(f"  Bands: {self.mode_data['band'].unique().tolist()}")
        print(f"  Defocal types: {self.mode_data['defocal_type'].value_counts().to_dict()}")

    def get_visit_ids(self) -> np.ndarray:
        """Return array of visit IDs for current mode."""
        return self.mode_data['visit'].values

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.mode_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return donut corresponding to the index."""
        # Get the row for this index
        row = self.mode_data.iloc[idx]

        # Get the image
        img = np.array(row['image'])

        # Get field position
        fx = float(row['field_x']) 
        fy = float(row['field_y']) 

        # Get the intra/extra flag
        intra = (row['defocal_type'] == 'intra')

        # Get the observed band - convert to index
        band_str = str(row['band'])
        band = "ugrizy".index(band_str)

        # Get Zernikes
        zernikes = np.array(row['zk_avg'])  # Ground truth (averaged)
        zk_individual = np.array(row['zk_individual'])  # Individual TIE for this donut
        
        # Get donut_blur
        if 'donut_blur' in row and not pd.isna(row['donut_blur']):
            donut_blur = float(row['donut_blur'])
        else:
            donut_blur = 1.0
        
        # Verify shapes
        if len(zernikes) != 21:
            if len(zernikes) < 21:
                zernikes = np.pad(zernikes, (0, 21 - len(zernikes)), constant_values=np.nan)
            else:
                zernikes = zernikes[:21]
        
        if len(zk_individual) != 21:
            if len(zk_individual) < 21:
                zk_individual = np.pad(zk_individual, (0, 21 - len(zk_individual)), constant_values=np.nan)
            else:
                zk_individual = zk_individual[:21]

        # Get visit ID
        visit_id = int(row['visit'])

        # Standardize inputs
        if self.settings["transform"]:
            img, fx, fy, intra, band = transform_inputs(
                img, fx, fy, intra, band
            )

        # Convert to tensors
        img = torch.from_numpy(img).float()
        fx = torch.FloatTensor([fx])
        fy = torch.FloatTensor([fy])
        intra = torch.FloatTensor([intra])
        band = torch.FloatTensor([band])
        zernikes = torch.from_numpy(zernikes.astype(np.float32)).float()
        zk_individual = torch.from_numpy(zk_individual.astype(np.float32)).float()  # ADD THIS
        donut_blur = torch.FloatTensor([donut_blur])

        output = {
            "image": img,
            "field_x": fx,
            "field_y": fy,
            "intrafocal": intra,
            "band": band,
            "zernikes": zernikes,
            "zk_individual": zk_individual, 
            "donut_blur": donut_blur,
            "visit_id": visit_id,
        }

        return output

