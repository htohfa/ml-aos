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
        """
        super().__init__()
        self.save_hyperparameters()
        self.wavenet = WaveNet(
            cnn_model=cnn_model,
            freeze_cnn=freeze_cnn,
            n_predictor_layers=n_predictor_layers,
        )

        # define some parameters that will be accessed by
        # the MachineLearningAlgorithm in ts_wep
        self.camType = "LsstCam"
        self.inputShape = (160, 160)

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
        zk_pred = convert_zernikes(zk_pred)
        zk_true = convert_zernikes(zk_true)

        mask = ~torch.isnan(zk_true)
        zk_pred = zk_pred[mask]
        zk_true = zk_true[mask]

        # pull out the weights from the final linear layer
        *_, A, _ = self.wavenet.predictor.parameters()

        # calculate Zernike loss
        sse = F.mse_loss(zk_pred, zk_true, reduction="none").sum(dim=-1)
        zk_loss = sse.mean() + self.hparams.alpha * A.square().sum()
        mRSSE = torch.sqrt(sse).mean()

        # calculate donut_blur loss (simple MSE)
        blur_loss = F.mse_loss(blur_pred.squeeze(), blur_true.squeeze())

        # combined loss
        loss = zk_loss + self.hparams.donut_blur_weight * blur_loss

        return loss, zk_loss, blur_loss, mRSSE

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute training step on a batch."""
        loss, zk_loss, blur_loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_zk_loss", zk_loss, sync_dist=True)
        self.log("train_blur_loss", blur_loss, sync_dist=True)
        self.log("train_mRSSE", mRSSE, sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute validation step on a batch."""
        loss, zk_loss, blur_loss, mRSSE = self.calc_losses(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_zk_loss", zk_loss, sync_dist=True)
        self.log("val_blur_loss", blur_loss, sync_dist=True)
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
        """Predict zernikes and donut_blur for production."""
        # transform the inputs
        from ml_aos.utils import transform_inputs

        img, fx, fy, focalFlag, band = transform_inputs(
            img, fx, fy, focalFlag, band
        )

        # predict zernikes and donut_blur
        zk_pred, blur_pred = self.wavenet(img, fx, fy, focalFlag, band)

        return zk_pred, blur_pred
