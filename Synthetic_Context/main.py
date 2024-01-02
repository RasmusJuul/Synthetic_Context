import argparse
import os
import sys

import torch

from src.models.train_model import main as train
from src.models.test_model import main as test
from src.models.train_cyclegan import main as train_cyclegan


def main(
    name: str,
    max_epochs: int,
    num_workers: int,
    lr: float,
    batch_size: int,
    compiled: bool,
    test_: bool,
    mix: bool,
    cyclegan: bool,
    seed: int,
    umap_subset: bool,
    pca_subset: bool,
):
    torch.cuda.empty_cache()
    if cyclegan:
        train_cyclegan(
            name=name,
            max_epochs=max_epochs,
            num_workers=num_workers,
            batch_size=batch_size,
            compiled=compiled,
            seed=seed,
        )
    elif test_:
        test(
            name=name, num_workers=num_workers, batch_size=batch_size, compiled=compiled
        )
    else:
        train(
            name=name,
            max_epochs=max_epochs,
            num_workers=num_workers,
            lr=lr,
            batch_size=batch_size,
            compiled=compiled,
            mix=mix,
            seed=seed,
            umap_subset=umap_subset,
            pca_subset=pca_subset,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a UNet on segmentation of BugNIST"
    )
    parser.add_argument("--name", "-n", type=str, default="test", help="Name for wandb")
    parser.add_argument(
        "--max-epochs", "-me", type=int, default=10, help="Number of max epochs"
    )
    parser.add_argument(
        "--num-workers",
        "-nw",
        type=int,
        default=0,
        help="Number of threads use in loading data",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=8,
    )
    parser.add_argument("-c", "--compiled", action="store_true", help="compiles model")
    parser.add_argument(
        "-t", "--test", action="store_true", help="if true test model else train"
    )
    parser.add_argument(
        "--mix",
        action="store_true",
        help="if true train on artificially created mixed samples",
    )
    parser.add_argument("--gan", action="store_true", help="if true train cyclegan")
    parser.add_argument(
        "--umap_subset",
        action="store_true",
        help="If True, uses a subset of the training data closest to real images in a UMAP (Uniform Manifold Approximation and Projection).")
    parser.add_argument(
        "--pca_subset",
        action="store_true",
        help="If True, uses a subset of the training data which are within the average distance between the real images when transformed using pca.")
    parser.add_argument(
        "--seed",
        type=int,
        help="seed for RNG",
        default=1234,
    )
    args = parser.parse_args()

    main(
        name=args.name,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        lr=args.lr,
        batch_size=args.batch_size,
        compiled=args.compiled,
        test_=args.test,
        mix=args.mix,
        cyclegan=args.gan,
        seed=args.seed,
        umap_subset=args.umap_subset,
        pca_subset=args.pca_subset,
    )
