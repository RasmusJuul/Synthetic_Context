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
            outputs.append(out['model.1.submodule.1.submodule.1.submodule.1.submodule.adn.A'].cpu())
    
    outputs = torch.concat(outputs,dim=0).flatten(start_dim=1)
    
    np.save(f"./notebooks/features/{name}_UNet_features.npy",outputs.numpy())
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the features of the deepest layer of the UNet trained on the 'tightly packed with noise' synthetic data. Pick which dataset you want the features of"
    )
    parser.add_argument(
        "--nn",
        action="store_true",
        help="tight packing no noise",
    )
    parser.add_argument(
        "--old",
        action="store_true",
        help="loose packing",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="real mixes",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="tight packing with noise",
    )
    args = parser.parse_args()

    if not args.nn ^ args.old ^ args.real ^ args.new:
        print("only one dataset can be picked")
    else:
    
        torch.set_float32_matmul_precision("medium")
    
        model = UNet_pl(
            spatial_dims=3,
            in_channels=1,
            out_channels=13,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
            lr=1,
        )
        
        # model_path = "../models/UNet_no_noise-2023-11-13-1713/UNet-epoch=343.ckpt"
        # model_path = "../models/UNet_old-2023-09-20-2208/UNet-epoch=229.ckpt"
        model_path = "../models/UNet-2023-11-03-0950/UNet-epoch=498.ckpt"
        model.load_state_dict(torch.load(model_path, map_location="cpu")['state_dict'], strict=True)
        
        torch._dynamo.config.suppress_errors = True
        
        feature_extractor = create_feature_extractor(model,['model.1.submodule.1.submodule.1.submodule.1.submodule.adn.A'])
        feature_extractor = torch.compile(feature_extractor)
        feature_extractor.eval();
        feature_extractor = feature_extractor.to("cuda");

        if args.nn or args.old or args.new:
            bugnist = BugNISTDataModule(batch_size=32, num_workers=16, mix=True,no_noise=args.nn, old=args.old)
            bugnist.setup(stage="test")
            dataloader = bugnist.test_dataloader()
        elif args.real:
            dataloader = DataLoader(MetricDataset(),
                             batch_size=32,
                             num_workers=16)

        if args.old:
            name = "loose_packing"
        elif args.nn:
            name = "tight_packing_nn"
        elif args.new:
            name = "tight_packing_wn"
        elif args.real:
            name = "real"
            
        extract_features(feature_extractor, dataloader,name)
