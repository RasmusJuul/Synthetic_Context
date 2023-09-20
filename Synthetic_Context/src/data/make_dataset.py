# -*- coding: utf-8 -*-
import logging
import os
from typing import Callable, List, Optional, Union

import PIL
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torchvision import transforms
from src import _PATH_DATA

import itertools
from enum import Enum, IntEnum, auto

import pandas as pd
from tifffile import tifffile
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import RandomRotation


class Label(IntEnum):
    Blowfly = 3,
    CurlyWingedFly = 6,
    Pupae = 10,
    Maggot = 8,
    BuffaloBeetleLarvae = 4,
    Mealworm = 9,
    SoliderFlyLarvae = 11,
    Woodlice = 12,
    BlackCricket = 2,
    Grasshopper = 7,
    BrownCricket = 1,
    BlowflyPupae = 5,

    @staticmethod
    def abbreviation_dict():
        assert len(Label) == 12
        return {
            Label.Blowfly: "BF",
            Label.CurlyWingedFly: "CF",
            Label.Pupae: "PP",
            Label.Maggot: "MA",
            Label.BuffaloBeetleLarvae: "BL",
            Label.Mealworm: "ML",
            Label.SoliderFlyLarvae: "SL",
            Label.Woodlice: "WO",
            Label.BlackCricket: "BC",
            Label.Grasshopper: "GH",
            Label.BrownCricket: "AC",
            Label.BlowflyPupae: "GP"
        }

    @property
    def abbreviation(self):
        return self.abbreviation_dict()[self]

    @staticmethod
    def from_abbreviation(abbreviation: str):
        return next(label for label, label_abbreviation in Label.abbreviation_dict().items() if label_abbreviation == abbreviation.upper())


class SplitType(Enum):
    Train = auto()
    Validation = auto()
    Test = auto()


class BugNIST_mix(torch.utils.data.Dataset):
    def __init__(self, type: SplitType, seed=42):

        dataset_path = _PATH_DATA

        self.image_paths, self.label_paths = self.dataset_images(dataset_path, type)
        self.image_paths = [os.path.join(dataset_path, path) for path in self.image_paths]
        self.label_paths = [os.path.join(dataset_path, path) for path in self.label_paths]
        self.rng = np.random.default_rng(seed=seed)

    @staticmethod
    def dataset_images(dataset_path: str, type: SplitType) -> tuple[list[str], list[Label]]:
        assert len(SplitType) == 3
        file_name = "train_mix" if type == SplitType.Train else "test_mix" if type == SplitType.Test else "validation_mix"
        files = pd.read_csv(f"{dataset_path}/{file_name}.csv", header=0)
        return files.image_path.to_list(), files.label_path.to_list()

    def __len__(self) -> int:
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = tifffile.imread(image_path)
        label = tifffile.imread(label_path)

        image = np.expand_dims(image, 0)
        
        X = torch.Tensor(image)
        y = torch.Tensor(label)
        y = y.to(torch.long)

        return X, y

    @staticmethod
    def num_classes() -> int:
        return len(Label)

    @staticmethod
    def label_to_name(label: int) -> str:
        return Label(label).abbreviation

    def get_name_of_image(self, idx: int) -> str:
        return self.image_paths[idx].split("/")[-1].split(".")[0]


class BugNIST(torch.utils.data.Dataset):
    def __init__(self, type: SplitType, seed=42):

        dataset_path = os.path.join(_PATH_DATA,"bugnist_512")

        self.image_paths, self.image_labels = self.dataset_images(dataset_path, type)
        self.image_paths = [os.path.join(dataset_path, path) for path in self.image_paths]
        self.rng = np.random.default_rng(seed=seed)
        self.rot = RandomRotation(180)

    @staticmethod
    def dataset_images(dataset_path: str, type: SplitType) -> tuple[list[str], list[Label]]:
        assert len(SplitType) == 3
        file_name = "train" if type == SplitType.Train else "test" if type == SplitType.Test else "validation"
        files = pd.read_csv(f"{dataset_path}/{file_name}.csv", names=["files"], header=0).files
        labels = [Label.from_abbreviation(abbreviation) for abbreviation in files.map(lambda x: x[:2]).to_list()]
        return files.to_list(), labels

    def __len__(self) -> int:
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[idx]
        label = self.image_labels[idx]

        image = tifffile.imread(image_path)

        image = np.expand_dims(image, 0)
        
        X = torch.Tensor(image)
        X = self.rot(X)

        target = torch.zeros(X.shape)
        # Mask to get the approximate area of the bug
        target[X >= 100] = label.value
        target = target.to(dtype=torch.long)
        target = target.squeeze(dim=0)
        
        # # One-hot encode
        # target = torch.nn.functional.one_hot(target, num_classes=13)
        # # Convert from NHWDC to NCHWD
        # target = target.permute(0, 4, 1, 2, 3)

        return X, target

    @staticmethod
    def num_classes() -> int:
        return len(Label)

    @staticmethod
    def label_to_name(label: int) -> str:
        return Label(label).abbreviation

    def get_name_of_image(self, idx: int) -> str:
        return self.image_paths[idx].split("/")[-1].split(".")[0]




class BugNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = _PATH_DATA, batch_size: int = 64, num_workers: int = 0, seed: int = 42, mix: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.mix = mix

    def setup(self, stage=None):        
        if stage == "test" or stage is None:
            if self.mix:
                self.bugnist_test = BugNIST_mix(type=SplitType.Test, seed=self.seed)
            else:
                self.bugnist_test = BugNIST(type=SplitType.Test, seed=self.seed)

        if stage == "fit" or stage is None:
            if self.mix:
                self.bugnist_train = BugNIST_mix(type=SplitType.Train, seed=self.seed)
                self.bugnist_val = BugNIST_mix(type=SplitType.Validation, seed=self.seed)
            else:
                self.bugnist_train = BugNIST(type=SplitType.Train, seed=self.seed)
                self.bugnist_val = BugNIST(type=SplitType.Validation, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(
            self.bugnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.bugnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.bugnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
