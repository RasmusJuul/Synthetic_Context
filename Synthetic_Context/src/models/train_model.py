import logging
import datetime

import torch
import torch._dynamo
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.data.dataloaders import BugNISTDataModule
from src.models.unet import UNet_pl


def main(
    name: str = "test",
    max_epochs: int = 10,
    num_workers: int = 0,
    lr: float = 1e-4,
    batch_size: int = 16,
    compiled: bool = False,
    mix: bool = False,
):
    seed_everything(1234, workers=True)

    time = str(datetime.datetime.now())[:-10].replace(" ", "-").replace(":", "")

    torch.set_float32_matmul_precision("medium")

    model = UNet_pl(
        spatial_dims=3,
        in_channels=1,
        out_channels=13,
        channels=(4, 8, 16, 32, 64),
        strides=(2, 2, 2, 2),
        lr=lr,
    )
    if compiled:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=_PATH_MODELS + "/" + name + "-" + time,
        filename="UNet-{epoch}",
        monitor="val/focal_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=True,
    )

    bugnist = BugNISTDataModule(batch_size=batch_size, num_workers=num_workers, mix=mix)

    wandb_logger = WandbLogger(project="Thesis", name=name)

    early_stopping_callback = EarlyStopping(
        monitor="val/focal_loss",
        patience=25,
        verbose=True,
        mode="min",
        strict=False,
        check_on_train_epoch_end=False,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        devices=-1,
        accelerator="gpu",
        deterministic=False,
        default_root_dir=_PROJECT_ROOT,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=25,
        logger=wandb_logger,
    )

    trainer.fit(
        model,
        datamodule=bugnist,
        # ckpt_path=_PATH_MODELS + "/2023-09-20-1155/UNet-epoch=164.ckpt",
    )

    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=bugnist)
