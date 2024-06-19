from pytorch_lightning import LightningModule
import torch
import torchmetrics as tm

from collections.abc import Sequence

from monai.networks.layers.factories import Act, Norm
from monai.networks.nets.unet import UNet

from src.utils.loss_functions import softmax_focal_loss, object_fill_loss

from monai.losses.dice import GeneralizedDiceLoss
from monai.networks import one_hot


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
        # self.alpha = torch.Tensor(
        #     [
        #         0.01,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #         0.0825,
        #     ]
        # )
        # self.alpha = torch.Tensor(
        #     [
        #         0.01,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #         0.04125,
        #     ]
        # )
        self.alpha = torch.tensor([1/out_channels]*out_channels)
        self.precision = tm.Precision(task="multiclass", num_classes=out_channels, ignore_index=0)
        self.recall = tm.Recall(task="multiclass", num_classes=out_channels, ignore_index=0)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y, centroids,_ = batch
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
        x, y, centroids,_ = batch
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
        x, y, centroids,_ = batch
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


class UNet_kaggle(UNet, LightningModule):
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
        super(UNet_kaggle, self).__init__(
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
        self.focal_loss = softmax_focal_loss
        self.alpha = torch.tensor([1/out_channels]*out_channels)
        
        self.object_fill_loss = object_fill_loss
        self.loss_weight = [10,1]

        
        self.lr = lr

        
        self.precision = tm.Precision(task="multiclass", num_classes=out_channels, ignore_index=0)
        self.recall = tm.Recall(task="multiclass", num_classes=out_channels, ignore_index=0)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y, centroids,objects = batch
        preds = self.forward(x)
        
        focal_loss = self.focal_loss(preds, y, alpha=self.alpha, reduction="mean")
        
        preds_ = preds.softmax(dim=1).argmax(dim=1)
        f_loss = self.object_fill_loss(preds_,objects)
        total_loss = self.loss_weight[0] * focal_loss + self.loss_weight[1] * f_loss
            
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)

        self.log_dict(
            {
                "train/focal_loss": focal_loss,
                "train/object_fill_loss": f_loss,
                "train/total_loss": total_loss,
                "train/precision": precision,
                "train/recall": recall,
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, centroids,objects = batch
        preds = self.forward(x)
        
        focal_loss = self.focal_loss(preds, y, alpha=self.alpha, reduction="mean")
        
        preds_ = preds.softmax(dim=1).argmax(dim=1)
        f_loss = self.object_fill_loss(preds_,objects)
        total_loss = self.loss_weight[0] * focal_loss + self.loss_weight[1] * f_loss
        
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)

        self.log_dict(
            {
                "val/focal_loss": focal_loss,
                "val/object_fill_loss": f_loss,
                "val/total_loss": total_loss,
                "val/precision": precision,
                "val/recall": recall,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y, centroids,objects = batch
        preds = self.forward(x)
        
        focal_loss = self.focal_loss(preds, y, alpha=self.alpha, reduction="mean")
        
        preds_ = preds.softmax(dim=1).argmax(dim=1)
        f_loss = self.object_fill_loss(preds_,objects)
        total_loss = self.loss_weight[0] * focal_loss + self.loss_weight[1] * f_loss
        
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)

        self.log_dict(
            {
                "test/focal_loss": focal_loss,
                "test/object_fill_loss": f_loss,
                "test/total_loss": total_loss,
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