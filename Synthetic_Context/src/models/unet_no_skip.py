from pytorch_lightning import LightningModule
import torch
import torchmetrics as tm
import torch.nn as nn

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
        
        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c #* 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c *2#+ channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, subblock, up_path)

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
