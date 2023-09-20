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
from pytorch_lightning.strategies import DeepSpeedStrategy

import wandb
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.data.make_dataset import BugNISTDataModule
from src.models.unet import UNet_pl


def main(
    name: str = "test",
    num_workers: int = 0,
    batch_size: int = 16,
    compiled: bool = False,
    ):
    
    seed_everything(1234, workers=True)
    
    time = str(datetime.datetime.now())[:-10].replace(" ","-").replace(":","")
    
    torch.set_float32_matmul_precision('medium')
    

    model=UNet_pl(spatial_dims=3,
                  in_channels=1,
                  out_channels=13,
                  channels=(4, 8, 16, 32, 64),
                  strides=(2, 2, 2, 2),
                )
    if compiled:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)

    bugnist = BugNISTDataModule(batch_size=batch_size, num_workers=num_workers,mix=True)

    wandb_logger = WandbLogger(project="Thesis", name=name)
    
    trainer = Trainer(
        max_epochs=1,
        devices=-1,
        accelerator="gpu",
        deterministic=False,
        default_root_dir=_PROJECT_ROOT,
        precision="16-mixed",
        log_every_n_steps=25,
        logger=wandb_logger,
    )
    
    trainer.test(model, datamodule=bugnist, ckpt_path="models/2023-09-19-1659/UNet-epoch=221.ckpt")
