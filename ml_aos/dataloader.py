"""Pytorch DataSet for the AOS simulations."""
import glob
from typing import Any, Dict

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import Dataset

from ml_aos.utils import transform_inputs


class Donuts(Dataset):
    """AOS donuts and zernikes from my simulations."""

    def __init__(
        self,
        mode: str = "train",
        transform: bool = True,
        fval: float = 0.1,
        ftest: float = 0.1,
        data_dir: str = "/astro/store/epyc/users/jfc20/data/aos_sims",
        seed: int = 0,
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
        fval: float, default=0.1
            Fraction of pointings in the validation set
        ftest: float, default=2048
            Fraction of pointings in the test set
        data_dir: str, default=/astro/store/epyc/users/jfc20/aos_sims
            Location of the data directory. The default location is where
            I stored the simulations on epyc.
        seed: int, default = 0
            Random seed for shuffling the data files.
        """
        # save the settings
        self.settings = {
            "mode": mode,
            "transform": transform,
            "data_dir": data_dir,
            "fval": fval,
            "ftest": ftest,
            "data_dir": data_dir,
            "seed": seed,
        }

        all_image_files = glob.glob(f"{data_dir}/images/*")

        # get a list of all the observations, and shuffle
        obs_ids = list(
            set([file.split("/")[-1].split(".")[1] for file in all_image_files])
        )
        rng = np.random.default_rng(seed)
        rng.shuffle(obs_ids)

        # separate the list of train, validation, and test observations
        frac_val = 0.1
        frac_test = 0.1
        n_val = int(frac_val * len(obs_ids))
        n_test = int(frac_test * len(obs_ids))

        self.obs_ids = {
            "train": obs_ids[n_val + n_test :],
            "val": obs_ids[:n_val],
            "test": obs_ids[n_val : n_val + n_test],
        }

        # partition the image files
        rng.shuffle(all_image_files)
        self.image_files = {
            mode: [
                file
                for file in all_image_files
                if file.split("/")[-1].split(".")[1] in ids
            ]
            for mode, ids in self.obs_ids.items()
        }

        # get the table of metadata for each observation
        self.observations = Table.read(f"{data_dir}/opSimTable.parquet")

    def __len__(self) -> int:
        """Return length of this Dataset."""
        return len(self.image_files[self.settings["mode"]])  # type: ignore

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
                image: donut image, shape=(256, 256)
                field_x, field_y: the field angle in radians
                intrafocal: boolean flag. 0 = extrafocal, 1 = intrafocal
                band: LSST band indicated by index in string "ugrizy" (e.g. 2 = "r")
                zernikes: Noll zernikes coefficients 4-21, inclusive (microns)
                dof: the telescope perturbations corresponding to the zernikes
                pntId: the pointing ID
                obsID: the observation ID
                objID: the object ID
        """
        # get the image file
        img_file = self.image_files[self.settings["mode"]][idx]  # type: ignore

        # load the image
        img = np.load(img_file)

        # crop out the central 160x160
        img = img[5:-5, 5:-5]

        # get the IDs
        pntId, obsId, objId = img_file.split("/")[-1].split(".")[:3]

        # get the catalog for this observation
        catalog = Table.read(
            f"{self.settings['data_dir']}/catalogs/{pntId}.catalog.parquet"
        )

        # get the row for this source
        row = catalog[catalog["objectId"] == int(objId[3:])][0]

        # get the donut locations
        fx, fy = row["xField"], row["yField"]

        # get the intra/extra flag
        intra = "SW1" in row["detector"]

        # get the observed band
        obs_row = self.observations[
            self.observations["observationId"] == int(obsId[3:])
        ]
        band = "ugrizy".index(obs_row["lsstFilter"].item())

        # load the zernikes
        zernikes = np.load(
            (
                f"{self.settings['data_dir']}/zernikes/"
                f"{pntId}.{obsId}.detector{row['detector'][:3]}.zernikes.npy"
            )
        )

        # load the degrees of freedom
        dof = np.load(f"{self.settings['data_dir']}/dof/{pntId}.dofs.npy")

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
            "pntId": int(pntId[3:]),
            "obsId": int(obsId[3:]),
            "objId": int(objId[3:]),
        }

        return output
