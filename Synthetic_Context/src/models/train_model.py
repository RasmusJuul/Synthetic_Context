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
from src.models.unet import UNet_pl, UNet_kaggle
from src.models.unet_no_skip import UNet_pl as UNetNoSkipConnection
from src.models.swin_unetr import SwinUNETR_pl as SwinUNETR


def main(
    name: str = "test",
    max_epochs: int = 10,
    num_workers: int = 0,
    lr: float = 1e-4,
    batch_size: int = 16,
    compiled: bool = False,
    mix: bool = False,
    seed: int = 1234,
    size: int = None,
    version: str = "v3",
    model: str = "small",
    umap_subset: bool = False,
    pca_subset: bool = False,
    feature_distance_subset: bool = False,
    gan_subset: bool = False,
    model_path: str = None,
):
    seed_everything(seed, workers=True)

    time = str(datetime.datetime.now())[:-10].replace(" ", "-").replace(":", "")

    torch.set_float32_matmul_precision("medium")

    if model == "small":
        model = UNet_pl(
            spatial_dims=3,
            in_channels=1,
            out_channels=13,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
            lr=lr,
        )
        filename="UNet_small-{epoch}"
        precision = "16-mixed"
    elif model == "large":
        if "kaggle" in version:
            model = UNet_kaggle(
                spatial_dims=3,
                in_channels=1,
                out_channels=13,
                channels=(16, 32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                num_res_units = 3,
                lr=lr,
            )
        else:
            model = UNet_pl(
                spatial_dims=3,
                in_channels=1,
                out_channels=13,
                channels=(16, 32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                num_res_units = 3,
                lr=lr,
            )
        filename="UNet_large-{epoch}"
        precision = "16-mixed"
    elif model == "swin":
        #model = SwinUNETR(img_size=(256,128,128), in_channels=1, out_channels=13, feature_size=24)
        model = SwinUNETR(img_size=(128,96,96), in_channels=1, out_channels=25, feature_size=48)
        filename="SwinUNETR-{epoch}"
        precision = "32-true"
        
    # model = UNetNoSkipConnection(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=13,
    #     channels=(16, 32, 64, 128, 256, 512),
    #     strides=(2, 2, 2, 2, 2),
    #     num_res_units = 3,
    #     lr=lr,
    # )
    

    if compiled:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{_PATH_MODELS}/{name}-{time}",
        filename=filename,
        monitor="val/total_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=True,
    )

    bugnist = BugNISTDataModule(batch_size=batch_size,
                                num_workers=num_workers,
                                mix=mix,
                                version=version,
                                size=size,
                                umap_subset=umap_subset,
                                pca_subset=pca_subset,
                                feature_distance_subset=feature_distance_subset,
                                gan=gan_subset,)

    wandb_logger = WandbLogger(project="Thesis", name=name)
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    if size == None:
        patience = 7
    elif size < 10000:
        patience = 18
    elif size > 10000:
        patience = 12

    early_stopping_callback = EarlyStopping(
        monitor="val/total_loss",
        patience=patience,
        verbose=True,
        mode="min",
        strict=False,
        check_on_train_epoch_end=False,
        check_finite = True,
    )

    
    trainer = Trainer(
        max_epochs=max_epochs,
        devices=-1,
        accelerator="gpu",
        deterministic=False,
        default_root_dir=_PROJECT_ROOT,
        precision=precision,
        callbacks=[checkpoint_callback, early_stopping_callback,lr_monitor],
        log_every_n_steps=25,
        logger=wandb_logger,
        strategy='ddp',
    )

    if model_path != None:
        trainer.fit(
            model,
            datamodule=bugnist,
            ckpt_path=f"{_PATH_MODELS}/{model_path}",
        )
    else:
        trainer.fit(
            model,
            datamodule=bugnist,
        )

    # trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=bugnist)
