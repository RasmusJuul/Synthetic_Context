from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
import torch
import torch._dynamo
from glob import glob
import pandas as pd
from tqdm import tqdm
import tifffile
from skimage import measure
import numpy as np

# Models
from src.models.unet import UNet_pl
from src.models.unet_no_skip import UNet_pl as UNetNoSkipConnection

# Feature extraction
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import umap

# Dataloaders
from src.data.dataloaders import BugNISTDataModule
from src.data.dataloaders import MetricDataset
from torch.utils.data import DataLoader
import argparse


def extract_features(feature_extractor, dataloader,name):
    outputs = []
    for img,_ in tqdm(dataloader, unit="batch"):
        img = img.to("cuda")
        with torch.no_grad():
            out = feature_extractor(img)
            outputs.append(out['model.1.1.1.1.1.add'].cpu())
    
    outputs = torch.concat(outputs,dim=0).flatten(start_dim=1)
    
    np.save(f"{_PROJECT_ROOT}/notebooks/features/{name}_UNetNoSkipConnection_features.npy",outputs.numpy())
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the features of the deepest layer of the UNet trained on the 'tightly packed with noise' synthetic data. Pick which dataset you want the features of"
    )
    parser.add_argument("--version",
                        type=str,
                        default="v3",
                        help="Which version of the dataset to use (v1, v2, or v3). Defaults to v3.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="real mixes",
    )
    parser.add_argument(
        "--gan",
        action="store_true",
        help="tight packing with noise",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="tight packing with noise",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=32,
    )
    args = parser.parse_args()


    
    torch.set_float32_matmul_precision("medium")

    # model = UNet_pl(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=13,
    #     channels=(4, 8, 16, 32, 64),
    #     strides=(2, 2, 2, 2),
    #     lr=1,
    # )
    
    # model = UNet_pl(
    # spatial_dims=3,
    # in_channels=1,
    # out_channels=13,
    # channels=(16, 32, 64, 128, 256, 512),
    # strides=(2, 2, 2, 2, 2),
    # num_res_units = 3,
    # )

    model = UNetNoSkipConnection(
    spatial_dims=3,
    in_channels=1,
    out_channels=13,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    num_res_units = 3,
)
    
    # model_path = "../models/UNet_no_noise-2023-11-13-1713/UNet-epoch=343.ckpt"
    # model_path = "../models/UNet_old-2023-09-20-2208/UNet-epoch=229.ckpt"
    # model_path = f"{_PATH_MODELS}/UNet-2023-11-03-0950/UNet-epoch=498.ckpt"
    # model_path = f"{_PATH_MODELS}/UNet_large-2023-12-06-1549/UNet-epoch=79.ckpt"
    model_path = f"{_PATH_MODELS}/UNet_no_skip-2024-01-08-1327/UNet-epoch=70.ckpt"
    
    model.load_state_dict(torch.load(model_path, map_location="cpu")['state_dict'], strict=True)
    
    torch._dynamo.config.suppress_errors = True
    
    # feature_extractor = create_feature_extractor(model,['model.1.submodule.1.submodule.1.submodule.1.submodule.adn.A']) # small model
    # feature_extractor = create_feature_extractor(model,['model.1.submodule.1.submodule.1.submodule.1.submodule.1.cat']) # large model
    feature_extractor = create_feature_extractor(model,['model.1.1.1.1.1.add']) # No skip connection model
    
    feature_extractor = torch.compile(feature_extractor)
    feature_extractor.eval();
    feature_extractor = feature_extractor.to("cuda");

    if args.real:
        dataloader = DataLoader(MetricDataset(),
                         batch_size=args.batch_size,
                         num_workers=16)
    else:
        bugnist = BugNISTDataModule(batch_size=args.batch_size, num_workers=16, mix=True, gan=args.gan, version=args.version)
        bugnist.setup()
        if args.train:
            dataloader = bugnist.train_dataloader(shuffle=False)
        else:
            dataloader = bugnist.test_dataloader()
    

    if args.version == "v1":
        name = "light_packing"
        if args.train:
            name = name+"_train"
        else:
            name = name+"_test"
    elif args.version == "v2":
        name = "dense_packing_nn"
        if args.train:
            name = name+"_train"
        else:
            name = name+"_test"
    elif args.version == "v3":
        name = "dense_packing_wn"
        if args.train:
            name = name+"_train"
        else:
            name = name+"_test"
            
    elif args.real:
        name = "real"
    elif args.gan:
        name = "gan"
        
    extract_features(feature_extractor, dataloader,name)
