# -*- coding: utf-8 -*-
import logging
import os
from typing import Callable, List, Optional, Union

import PIL
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src import _PATH_DATA
from glob import glob

import itertools
from enum import Enum, IntEnum, auto

import pandas as pd
from skimage.transform import resize
from tifffile import tifffile
from torch import Tensor
from torch.utils.data import DataLoader
from monai import transforms


class ImageFolder:
    def __init__(self, root):
        self.root = root

        self.paths = glob("**/*.tif", root_dir=root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):  # doesnt support slices, we dont want them
        idx = idx % len(self)
        return os.path.join(self.root, self.paths[idx]), tifffile.imread(
            os.path.join(self.root, self.paths[idx])
        )


class UnpairedDataset(Dataset):
    def __init__(self, root: str = "cyclegan_256", mode: str = "train"):
        """
        root must have trainA trainB testA testB as its subfolders
        mode must be either 'train', 'test' or 'validation'
        """
        assert (
            mode in "train test validation".split()
        ), "mode should be either train, test or validation"

        super().__init__()

        root = os.path.join(_PATH_DATA, root)
        # pathA = os.path.join(_PATH_DATA,f"synthetic_mixed_256/{mode}")
        pathA = os.path.join(root, mode + "A")
        self.dirA = ImageFolder(pathA)

        pathB = os.path.join(root, mode + "B")
        self.dirB = ImageFolder(pathB)

        print(
            f"Found {len(self.dirA)} images of {mode}A and {len(self.dirB)} images of {mode}B"
        )

    def __len__(self):
        return max(len(self.dirA), len(self.dirB))

    def load_image(self, path):
        image = tifffile.imread(path)
        image = torch.Tensor(image, dtype=torch.float)
        return path, image

    def __getitem__(self, idx):  # doesnt support slices, we dont want them
        # we use serial batching
        pathA, imgA = self.dirA[idx]
        pathB, imgB = self.dirB[idx]

        imgA = torch.Tensor(imgA)
        imgB = torch.Tensor(imgB)

        imgA = imgA.unsqueeze(dim=0)
        imgB = imgB.unsqueeze(dim=0)

        imgA = imgA.to(torch.float)
        imgB = imgB.to(torch.float)

        return {"A": imgA, "pathA": pathA, "B": imgB, "pathB": pathB}


class CycleGANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = _PATH_DATA,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        if stage == "test" or stage is None:
            self.cyclegan_test = UnpairedDataset(mode="test")

        if stage == "fit" or stage is None:
            self.cyclegan_train = UnpairedDataset(mode="train")
            self.cyclegan_val = UnpairedDataset(mode="validation")

    def train_dataloader(self):
        return DataLoader(
            self.cyclegan_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cyclegan_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cyclegan_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

class MetricDataset(Dataset):
    def __init__(self):
        """
        Dataset of mixed_crop and their corresponding metric distance labels
        """
        super().__init__()

        files = pd.read_csv(f"{_PATH_DATA}/mixed_and_label_paths.csv", header=0)

        self.image_paths = files.img_path.to_list()
        self.label_paths = files.label_path.to_list()
        
        self.image_paths = [
            os.path.join(_PATH_DATA, path) for path in self.image_paths
        ]
        self.label_paths = [
            os.path.join(_PATH_DATA, path) for path in self.label_paths
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

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

    def get_name_of_image(self, idx: int) -> str:
        return self.image_paths[idx].split("/")[-1].split(".")[0]
        

class Label(IntEnum):
    Blowfly = (3,)
    CurlyWingedFly = (6,)
    Pupae = (10,)
    Maggot = (8,)
    BuffaloBeetleLarvae = (4,)
    Mealworm = (9,)
    SoliderFlyLarvae = (11,)
    Woodlice = (12,)
    BlackCricket = (2,)
    Grasshopper = (7,)
    BrownCricket = (1,)
    BlowflyPupae = (5,)

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
            Label.BlowflyPupae: "GP",
        }

    @property
    def abbreviation(self):
        return self.abbreviation_dict()[self]

    @staticmethod
    def from_abbreviation(abbreviation: str):
        return next(
            label
            for label, label_abbreviation in Label.abbreviation_dict().items()
            if label_abbreviation == abbreviation.upper()
        )


class SplitType(Enum):
    Train = auto()
    Validation = auto()
    Test = auto()

# Dataloader for images of synthetic mixes or synthetic mixes generated with a cyclegan
class BugNIST_mix(torch.utils.data.Dataset):
    def __init__(self, type: SplitType, seed=42, gan=False, transform=False,no_noise=False,old=False):
        dataset_path = _PATH_DATA

        self.image_paths, self.label_paths = self.dataset_images(dataset_path, type, gan, no_noise,old)
        self.image_paths = [
            os.path.join(dataset_path, path) for path in self.image_paths
        ]
        self.label_paths = [
            os.path.join(dataset_path, path) for path in self.label_paths
        ]
        self.rng = np.random.default_rng(seed=seed)
        
        self.transforms_img = transforms.Compose([transforms.RandFlip(prob=0.5,spatial_axis=2),
                                                  transforms.RandFlip(prob=0.5,spatial_axis=1),
                                                  transforms.RandFlip(prob=0.5,spatial_axis=0),
                                                  transforms.RandRotate90(prob=0.3,spatial_axes=(1,2)),
                                                  ])
        
        self.transforms_label = transforms.Compose([transforms.RandFlip(prob=0.5,spatial_axis=2),
                                                    transforms.RandFlip(prob=0.5,spatial_axis=1),
                                                    transforms.RandFlip(prob=0.5,spatial_axis=0),
                                                    transforms.RandRotate90(prob=0.3,spatial_axes=(1,2)),
                                                    ])
        
        self.transforms_img = self.transforms_img.set_random_state(seed=seed)
        self.transforms_label = self.transforms_label.set_random_state(seed=seed)
        
        self.transform = transform

    @staticmethod
    def dataset_images(
        dataset_path: str, type: SplitType, gan: bool, no_noise: bool, old: bool,
    ) -> tuple[list[str], list[str]]:
        assert len(SplitType) == 3
        # file_name = "train_mix" if type == SplitType.Train else "test_mix" if type == SplitType.Test else "validation_mix"
        file_name = (
            "train"
            if type == SplitType.Train
            else "test"
            if type == SplitType.Test
            else "validation"
        )
        if no_noise:
            file_name += "_no_noise"
        elif old:
            file_name += "_old"
        files = pd.read_csv(f"{dataset_path}/{file_name}.csv", header=0)
        if gan:
            return files.gan_img_path.to_list(), files.label_path.to_list()
        else:
            return files.img_path.to_list(), files.label_path.to_list()

    def __len__(self) -> int:
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = tifffile.imread(image_path)
        label = tifffile.imread(label_path)

        X = torch.Tensor(image)
        X = X.unsqueeze(dim=0)
        y = torch.Tensor(label)
        y = y.unsqueeze(dim=0)
        
        if self.transform:
            X = self.transforms_img(X)
            y = self.transforms_label(y)

        X = X.to(torch.float32)
        y = y.squeeze(dim=0)
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

# Dataloader for images of single insects and synthetic mixes
class BugNIST_all(torch.utils.data.Dataset):
    def __init__(self, type: SplitType, seed=42):
        dataset_path = _PATH_DATA

        self.image_paths, self.label_paths = self.dataset_images(dataset_path, type)
        self.image_paths = [
            os.path.join(dataset_path, path) for path in self.image_paths
        ]
        self.label_paths = [
            os.path.join(dataset_path, path) for path in self.label_paths
        ]
        self.rng = np.random.default_rng(seed=seed)

    @staticmethod
    def dataset_images(
        dataset_path: str, type: SplitType
    ) -> tuple[list[str], list[Label]]:
        assert len(SplitType) == 3
        file_name = (
            "train"
            if type == SplitType.Train
            else "test"
            if type == SplitType.Test
            else "validation"
        )
        files = pd.read_csv(f"{dataset_path}/{file_name}.csv", header=0)
        return files.image_path.to_list(), files.label_path.to_list()

    def __len__(self) -> int:
        return len(self.image_paths)  # len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = tifffile.imread(image_path)
        image = np.expand_dims(image, 0)

        X = torch.Tensor(image)

        if "none" in label_path:
            target = torch.zeros(X.shape)
            # Mask to get the approximate area of the bug
            target[X >= 100] = Label.from_abbreviation(image_path.split("/")[-2])
            target = target.to(dtype=torch.long)
            y = target.squeeze(dim=0)

        else:
            label = tifffile.imread(label_path)
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

# Dataloader for images of single insects
class BugNIST(torch.utils.data.Dataset):
    def __init__(self, type: SplitType, seed=42):
        dataset_path = os.path.join(_PATH_DATA, "bugnist_512")

        self.image_paths, self.image_labels = self.dataset_images(dataset_path, type)
        self.image_paths = [
            os.path.join(dataset_path, path) for path in self.image_paths
        ]
        self.rng = np.random.default_rng(seed=seed)
        self.rot = RandomRotation(180)

    @staticmethod
    def dataset_images(
        dataset_path: str, type: SplitType
    ) -> tuple[list[str], list[Label]]:
        assert len(SplitType) == 3
        file_name = (
            "train"
            if type == SplitType.Train
            else "test"
            if type == SplitType.Test
            else "validation"
        )
        files = pd.read_csv(
            f"{dataset_path}/{file_name}.csv", names=["files"], header=0
        ).files
        labels = [
            Label.from_abbreviation(abbreviation)
            for abbreviation in files.map(lambda x: x[:2]).to_list()
        ]
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
        self,
        data_dir: str = _PATH_DATA,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 42,
        mix: bool = False,
        no_noise: bool = False,
        old: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.mix = mix
        self.no_noise = no_noise
        self.old = old

    def setup(self, stage=None):
        if stage == "test" or stage is None:
            if self.mix:
                self.bugnist_test = BugNIST_mix(type=SplitType.Test, seed=self.seed, no_noise=self.no_noise, old=self.old, gan=False)
            else:
                self.bugnist_test = BugNIST(type=SplitType.Test, seed=self.seed)

        if stage == "fit" or stage is None:
            if self.mix:
                self.bugnist_train = BugNIST_mix(type=SplitType.Train, seed=self.seed, no_noise=self.no_noise, old=self.old, gan=False,transform=True)
                self.bugnist_val = BugNIST_mix(
                    type=SplitType.Validation, seed=self.seed, no_noise=self.no_noise, old=self.old, gan=False,
                )
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
