from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.data.dataloaders import MetricDataset#, BugNISTDataModule
from src.models.unet import UNet_pl
from src.models.swin_unetr import SwinUNETR_pl as SwinUNETR
import torch
import torch._dynamo
from glob import glob
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile
from skimage import measure
import numpy as np
import argparse

def test(model, test_loader):
    
    color_dict = {1:[255,0,0],2:[0,255,0],3:[0,0,255],4:[255,255,0],5:[255,0,255],6:[0,255,255],7:[161,161,255],8:[171,128,84],9:[255,128,191],10:[135,89,179],11:[255,191,128],12:[0,85,0]}
    accuracies = []
    for k,(img,label) in enumerate(tqdm(test_loader, unit="batch")):
        img = img.to("cuda")
        with torch.no_grad():
            preds = model(img);
            preds_sm = preds.softmax(dim=1).cpu().numpy();
    
            class_props = []
            for i in range(1, 13):  # For each class channel
                props = measure.regionprops(
            
                    #                        v-- softmax channel for this class
            
                    measure.label(preds_sm[0, i] > 0.25)  # Threshold softmax probability at 0.25
            
                )
            
                props = [p for p in props if p.area > 5**3]  # Remove small connected components
            
                class_props.append(props)
            
            for i, props in enumerate(class_props):
                # For every found connected component
                for p in props:
                    bb = p.bbox
            
                    # Sets the found connected component to the mean value of the connected component
                    preds_sm[0,i+1,bb[0]:bb[3],bb[1]:bb[4],bb[2]:bb[5]][p.image] = preds_sm[0,i+1,bb[0]:bb[3],bb[1]:bb[4],bb[2]:bb[5]][p.image].mean()
            
            out = torch.Tensor(preds_sm).softmax(dim=1).argmax(dim=1).to(torch.uint8)
            accuracies.append(sum(out[out != 0] == label[out != 0])/len(label[out != 0]))
            
    accuracies = torch.Tensor(accuracies)
    print(accuracies.mean())
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--size", type=str, default="5000", help="model param path")
    parser.add_argument("--model", type=str, default="small", help="model type")
    parser.add_argument("--version", type=str, default="v1", help="which version of the dataset")
    parser.add_argument("--fixed", action="store_true", help="test on the 18 manually fixed labels")
    

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    test_loader = DataLoader(MetricDataset(fixed=args.fixed),
                         batch_size=1,
                         num_workers=0)

    if args.model == "small":
        model = UNet_pl(
            spatial_dims=3,
            in_channels=1,
            out_channels=13,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
            lr=1,
        )
    elif args.model == "large":
        model = UNet_pl(
                spatial_dims=3,
                in_channels=1,
                out_channels=13,
                channels=(16, 32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                num_res_units = 3,
            )
    elif args.model == "swin":
        model = SwinUNETR(img_size=(256,128,128), in_channels=1, out_channels=13, feature_size=24)

    
    if args.version == "single":
        model_path = glob(f"models/{args.model}_{args.version}*/*.ckpt")[0]
    elif args.version == "fd":
        model_path = glob(f"models/{args.model}_{args.version}*/*.ckpt")[0]
    elif args.version == "umap":
        model_path = glob(f"models/{args.model}_{args.version}*/*.ckpt")[0]
    elif args.version == "pca":
        model_path = glob(f"models/{args.model}_{args.version}*/*.ckpt")[0]
    elif args.version == "gan":
        model_path = glob(f"models/{args.model}_{args.version}*/*.ckpt")[0]
    else:
        model_path = glob(f"models/{args.model}{args.size}{args.version}*/*.ckpt")[0]
    model.load_state_dict(torch.load(model_path, map_location=None)['state_dict'], strict=True)
        
    torch._dynamo.config.suppress_errors = True
    if args.model != "swin":
        model = torch.compile(model)
    model.eval();
    model = model.to("cuda")

    test(model,test_loader)
    print(model_path)