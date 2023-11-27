from pytorch_lightning import LightningModule
import torch
import torchmetrics as tm

from collections.abc import Sequence

from monai.networks.layers.factories import Act, Norm
from monai.networks.nets.unet import UNet

from src.utils.loss_functions import softmax_focal_loss


class UNet_pl(UNet, LightningModule):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        lr: float = 1e-3,
    ) -> None:
        super(UNet_pl, self).__init__(
            spatial_dims,
            in_channels,
            out_channels,
            channels,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            act,
            norm,
            dropout,
            bias,
            adn_ordering,
        )
        self.Loss = softmax_focal_loss
        self.lr = lr
        self.alpha = torch.Tensor(
            [
                0.01,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
                0.0825,
            ]
        )
        self.precision = tm.Precision(task="multiclass", num_classes=13, ignore_index=0)
        self.recall = tm.Recall(task="multiclass", num_classes=13, ignore_index=0)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        preds = self.forward(x)
        loss = self.Loss(preds, y, alpha=self.alpha, reduction="mean")
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)

        self.log_dict(
            {
                "train/focal_loss": loss,
                "train/precision": precision,
                "train/recall": recall,
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.Loss(preds, y, alpha=self.alpha, reduction="mean")
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)

        self.log_dict(
            {
                "val/focal_loss": loss,
                "val/precision": precision,
                "val/recall": recall,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.Loss(preds, y, alpha=self.alpha, reduction="mean")
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)

        self.log_dict(
            {
                "test/focal_loss": loss,
                "test/precision": precision,
                "test/recall": recall,
            },
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, cooldown=2)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "val/focal_loss",}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
