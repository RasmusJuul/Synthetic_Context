from pytorch_lightning import LightningModule
import torch
import torchmetrics as tm

from collections.abc import Sequence

from monai.networks.layers.factories import Act, Norm
from monai.networks.nets.swin_unetr import SwinUNETR

from src.utils.loss_functions import softmax_focal_loss


class SwinUNETR_pl(SwinUNETR, LightningModule):
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        dropout_path_rate: float = 0.1,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=True,
        lr: float = 8e-4,
    ) -> None:
        super(SwinUNETR_pl, self).__init__(
            img_size,
            in_channels,
            out_channels,
            depths,
            num_heads,
            feature_size,
            norm_name,
            drop_rate,
            attn_drop_rate,
            dropout_path_rate,
            normalize,
            use_checkpoint,
            spatial_dims,
            downsample,
            use_v2,
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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, cooldown=1)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "val/focal_loss",}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
