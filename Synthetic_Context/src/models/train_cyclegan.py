import logging
import datetime

import torch
import torch._dynamo
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.data.dataloaders import CycleGANDataModule
from src.models.CycleGan import CycleGan


def main(
    name: str = "test",
    max_epochs: int = 10,
    num_workers: int = 0,
    lr: float = 1e-4,
    batch_size: int = 16,
    compiled: bool = False,
):
    seed_everything(1234, workers=True)

    time = str(datetime.datetime.now())[:-10].replace(" ", "-").replace(":", "")

    torch.set_float32_matmul_precision("medium")

    model = CycleGan()

    if compiled:
        # torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=_PATH_MODELS + "/" + name + "-" + time,
        filename="CycleGAN-{epoch}",
        every_n_epochs=1,
        save_top_k=-1,
        auto_insert_metric_name=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    datamodule = CycleGANDataModule(batch_size=batch_size, num_workers=num_workers)

    wandb_logger = WandbLogger(project="Thesis", name=name)

    trainer = Trainer(
        max_epochs=max_epochs,
        devices=-1,
        accelerator="gpu",
        deterministic=False,
        default_root_dir=_PROJECT_ROOT,
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=25,
        logger=wandb_logger,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=_PATH_MODELS + "/2023-09-27-2348/CycleGAN-epoch=3.ckpt",
    )
