"""PyTorch Lightning modules for training WaveNet."""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml_aos.real_data_loader import RealDonuts
from ml_aos.utils import convert_zernikes
from ml_aos.wavenet import WaveNet


class RealDonutLoader(pl.LightningDataModule):
    """DataModule for real commissioning data."""

    def __init__(
        self,
        data_dir: str = "/astro/store/epyc/ssd/users/htohfa/aos_chunks",
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        filter_low_snr: bool = False,
        snr_percentile: float = 20.0,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        """Create the RealDonutLoader."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.filter_low_snr = filter_low_snr
        self.snr_percentile = snr_percentile
        self.random_seed = random_seed
        self.kwargs = kwargs

    def setup(self, stage=None):
        """Set up the datasets."""
        self.train_dataset = RealDonuts(
            mode="train",
            data_dir=self.data_dir,
            filter_low_snr=self.filter_low_snr,
            snr_percentile=self.snr_percentile,
            random_seed=self.random_seed,
            **self.kwargs,
        )
        self.val_dataset = RealDonuts(
            mode="val",
            data_dir=self.data_dir,
            filter_low_snr=self.filter_low_snr,
            snr_percentile=self.snr_percentile,
            random_seed=self.random_seed,
            **self.kwargs,
        )
        self.test_dataset = RealDonuts(
            mode="test",
            data_dir=self.data_dir,
            filter_low_snr=self.filter_low_snr,
            snr_percentile=self.snr_percentile,
            random_seed=self.random_seed,
            **self.kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )


class WaveNetSystem(pl.LightningModule):
    """Pytorch Lightning system for training the WaveNet."""

    def __init__(
        self,
        cnn_model: str = "resnet18",
        freeze_cnn: bool = False,
        n_predictor_layers: tuple = (256,),
        alpha: float = 0,
        lr: float = 1e-3,
        lr_schedule: bool = False,
        donut_blur_weight: float = 0.1,
        defocus_weight: float = 2.0,  
    ) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        cnn_model: str, default="resnet18"
            The name of the pre-trained CNN model from torchvision.
        freeze_cnn: bool, default=False
            Whether to freeze the CNN weights.
        n_predictor_layers: tuple, default=(256)
            Number of nodes in the hidden layers of the predictor network.
        alpha: float, default=0
            Weight for the L2 penalty.
        lr: float, default=1e-3
            The initial learning rate for Adam.
        lr_schedule: bool, default=False
            Whether to use the ReduceLROnPlateau learning rate scheduler.
        donut_blur_weight: float, default=0.1
            Weight for donut_blur loss relative to Zernike loss.
        defocus_weight: float, default=2.0
            Weight for defocus (Z4) loss. Set to 0.0 to disable.
        """
        super().__init__()
        self.save_hyperparameters()
        self.wavenet = WaveNet(
            cnn_model=cnn_model,
            freeze_cnn=freeze_cnn,
            n_predictor_layers=n_predictor_layers,
        )

        # save Noll indices for output Zernikes
        # (note type annotation is required for torchscript export)
        self.nollIndices: torch.Tensor = torch.tensor(
            list(range(4, 20)) + list(range(22, 27))
        )

    def predict_step(
        self, batch: dict, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict Zernikes and donut_blur, return with truth."""
        # unpack data from the dictionary
        img = batch["image"]
        fx = batch["field_x"]
        fy = batch["field_y"]
        intra = batch["intrafocal"]
        band = batch["band"]
        zk_true = batch["zernikes"]
        blur_true = batch["donut_blur"]

        # predict zernikes and donut_blur
        zk_pred, blur_pred = self.wavenet(img, fx, fy, intra, band)

        return zk_pred, zk_true, blur_pred, blur_true

    def calc_losses(self, batch: dict, batch_idx: int) -> tuple:
        """Predict Zernikes and donut_blur, calculate the losses."""
        # predict zernikes and donut_blur
        zk_pred, zk_true, blur_pred, blur_true = self.predict_step(batch, batch_idx)

        # convert Zernikes to FWHM contributions
        zk_pred_weighted = convert_zernikes(zk_pred)
        zk_true_weighted = convert_zernikes(zk_true)

        mask = ~torch.isnan(zk_true_weighted)
        zk_pred_masked = zk_pred_weighted[mask]
        zk_true_masked = zk_true_weighted[mask]

        # pull out the weights from the final linear layer
        *_, A, _ = self.wavenet.predictor.parameters()

        # calculate Zernike loss
        sse = F.mse_loss(zk_pred_masked, zk_true_masked, reduction="none").sum(dim=-1)
        zk_loss = sse.mean() + self.hparams.alpha * A.square().sum()
        mRSSE = torch.sqrt(sse).mean()

        # calculate defocus loss (index 0 = Z4)
        defocus_pred = zk_pred_weighted[:, 0]
        defocus_true = zk_true_weighted[:, 0]
        defocus_mask = ~torch.isnan(defocus_true)
        defocus_loss = F.mse_loss(defocus_pred[defocus_mask], defocus_true[defocus_mask])

        # calculate donut_blur loss (simple MSE)
        blur_loss = F.mse_loss(blur_pred.squeeze(), blur_true.squeeze())

        # combined loss
        loss = zk_loss + self.hparams.donut_blur_weight * blur_loss + self.hparams.defocus_weight * defocus_loss

        return loss, zk_loss, blur_loss, defocus_loss, mRSSE

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute training step on a batch."""
        loss, zk_loss, blur_loss, defocus_loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_zk_loss", zk_loss, sync_dist=True)
        self.log("train_blur_loss", blur_loss, sync_dist=True)
        self.log("train_defocus_loss", defocus_loss, sync_dist=True)
        self.log("train_mRSSE", mRSSE, sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        loss, zk_loss, blur_loss, defocus_loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_zk_loss", zk_loss, sync_dist=True)
        self.log("val_blur_loss", blur_loss, sync_dist=True)
        self.log("val_defocus_loss", defocus_loss, sync_dist=True)
        self.log("val_mRSSE", mRSSE, sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        if self.hparams.lr_schedule:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer),
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def forward(
        self,
        img: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        focalFlag: torch.Tensor,
        band: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict zernikes and donut_blur for production.

        This method assumes the inputs have NOT been previously
        transformed by ml_aos.utils.transform_inputs.
        """
        # rescale image to [0, 1]
        img -= img.min()
        img /= img.max()

        # normalize image
        image_mean = 0.347
        image_std = 0.226
        img = (img - image_mean) / image_std

        # convert angles to radians
        fx *= torch.pi / 180
        fy *= torch.pi / 180

        # normalize angles
        field_mean = 0.000
        field_std = 0.021
        fx = (fx - field_mean) / field_std
        fy = (fy - field_mean) / field_std

        # normalize the intrafocal flags
        intra_mean = 0.5
        intra_std = 0.5
        focalFlag = (focalFlag - intra_mean) / intra_std

        # get the effective wavelength in microns
        # Map band integers to their corresponding scalar values
        band_map = torch.tensor(
            [0.3671, 0.4827, 0.6223, 0.7546, 0.8691, 0.9712],
            device=band.device,
        )
        band = band_map[band.long().squeeze() - 1].unsqueeze(1)

        # normalize the wavelength
        band_mean = 0.710
        band_std = 0.174
        band = (band - band_mean) / band_std

        # predict zernikes in microns, and donut blur in arcseconds
        zk_pred, blur_pred = self.wavenet(img, fx, fy, focalFlag, band)

        # convert to meters
        zk_pred /= 1e6

        # remove the zernikes that were not trained on real data
        idx = self.nollIndices - 4
        zk_pred = zk_pred[:, idx]

        blur_pred = blur_pred.squeeze()

        return zk_pred, blur_pred
