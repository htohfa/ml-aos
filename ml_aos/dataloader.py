
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict

from ml_aos.utils import transform_inputs
import sys
#sys.path.append('/content/ml-aos')
from ml_aos.load_data import load_data


class Donuts(Dataset):
    """AOS donuts and zernikes from my simulations."""

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the simulated AOS donuts and zernikes in a Pytorch Dataset.

        Parameters
        ----------
        mode: str, default="train"
            Which set to load. Options are train, val (i.e. validation),
            or test.
        transform: bool, default=True
            Whether to apply transform_inputs from ml_aos.utils.
        """
        # save the settings
        self.settings = {
            "mode": mode,
            "transform": transform,
        }

        # Load all data using your load_data function
        self.data = load_data()
        
        # Get unique observation IDs for splitting
        obs_ids = list(self.data['obsId'].unique())
        
        # Get unique bands for splitting
        unique_bands = self.data['filter'].unique()
        
        # Split the observations between train, test, val
        train_ids = []
        val_ids = []
        test_ids = []

        # Handle each band separately
        for band in unique_bands:
            band_obs = self.data[self.data['filter'] == band]['obsId'].unique()
            
            if band == 'u':
                # For u band: 2 in test, rest in train (if there are fewer observations)
                if len(band_obs) >= 2:
                    test_ids.extend(band_obs[:2])
                    train_ids.extend(band_obs[2:])
                else:
                    train_ids.extend(band_obs)
            else:
                # For other bands: 2 in test, 2 in val, rest in train
                if len(band_obs) >= 4:
                    test_ids.extend(band_obs[:2])
                    val_ids.extend(band_obs[2:4])
                    train_ids.extend(band_obs[4:])
                elif len(band_obs) >= 2:
                    test_ids.extend(band_obs[:1])
                    val_ids.extend(band_obs[1:2])
                    train_ids.extend(band_obs[2:])
                else:
                    train_ids.extend(band_obs)

        self.obs_ids = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        # Filter data for the current mode
        mode_obs_ids = self.obs_ids[mode]
        self.mode_data = self.data[self.data['obsId'].isin(mode_obs_ids)].reset_index(drop=True)

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.mode_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return simulation corresponding to the index.

        Parameters
        ----------
        idx: int
            The index of the simulation to return.

        Returns
        -------
        dict
            The dictionary contains the following pytorch tensors
                image: donut image, shape=(160, 160) after cropping
                field_x, field_y: the field angle in radians
                intrafocal: boolean flag. 0 = extrafocal, 1 = intrafocal
                band: LSST band indicated by index in string "ugrizy" (e.g. 2 = "r")
                zernikes: Noll zernikes coefficients 4-21, inclusive (microns)
                dof: the telescope perturbations corresponding to the zernikes
                pntId: the pointing ID
                obsID: the observation ID
                objID: the object ID
        """
        # Get the row for this index
        row = self.mode_data.iloc[idx]
        
        # Get the image - assuming it's stored in the 'image' column
        img = row['image']
        if isinstance(img, str):
            # If image is a file path, load it
            img = np.load(img)
        else:
            # If image is already an array
            img = np.array(img)
        
        # Crop out the central 160x160 (assuming original is 170x170 based on your config)
        if img.shape[0] > 160:
            crop_size = (img.shape[0] - 160) // 2
            img = img[crop_size:-crop_size, crop_size:-crop_size]

        # Get field coordinates
        fx = float(row['fx'])
        fy = float(row['fy'])

        # Get the intra/extra flag
        intra = bool(row['intra'])

        # Get the observed band - convert filter name to index
        band_str = row['filter']
        band = "ugrizy".index(band_str)

        # Get zernikes - assuming they're in columns or can be extracted
        # You might need to adjust this based on how zernikes are stored in your data
        if 'zernikes' in row:
            zernikes = np.array(row['zernikes'])
        else:
            # If zernikes are in separate columns, collect them
            zernike_cols = [col for col in self.mode_data.columns if 'zernike' in col.lower()]
            if zernike_cols:
                zernikes = np.array([row[col] for col in zernike_cols])
            else:
                # Default to zeros if no zernikes found
                zernikes = np.zeros(18)  # Zernikes 4-21 inclusive

        # Get degrees of freedom
        if 'dof' in row:
            dof = np.array(row['dof'])
        else:
            # If dof are in separate columns, collect them
            dof_cols = [col for col in self.mode_data.columns if 'dof' in col.lower()]
            if dof_cols:
                dof = np.array([row[col] for col in dof_cols])
            else:
                # Default to zeros if no dof found
                dof = np.zeros(50)  # Adjust size as needed

        # Get IDs
        pntId = int(row['pntId'])
        obsId = int(row['obsId'])
        objId = int(row['objId'])

        # standardize all the inputs for the neural net
        if self.settings["transform"]:
            img, fx, fy, intra, band = transform_inputs(
                img,
                fx,
                fy,
                intra,
                band,
            )

        # convert everything to tensors
        img = torch.from_numpy(img).float()
        fx = torch.FloatTensor([fx])
        fy = torch.FloatTensor([fy])
        intra = torch.FloatTensor([intra])
        band = torch.FloatTensor([band])
        zernikes = torch.from_numpy(zernikes).float()
        dof = torch.from_numpy(dof).float()

        output = {
            "image": img,
            "field_x": fx,
            "field_y": fy,
            "intrafocal": intra,
            "band": band,
            "zernikes": zernikes,
            "dof": dof,
            "pntId": pntId,
            "obsId": obsId,
            "objId": objId,
        }

        return output
