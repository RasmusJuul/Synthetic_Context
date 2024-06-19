from glob import glob
import pandas as pd
import numpy as np
from src import _PATH_DATA
import argparse


def create_csv(split,version,gan):
    folder_name = f"synthetic_mixed_{version}"
        
    img_path = glob(f"{folder_name}/{split}/**/mix*.tif",root_dir=_PATH_DATA)
    label_path = glob(f"{folder_name}/{split}/**/label*.tif",root_dir=_PATH_DATA)
    centroid_path = glob(f"{folder_name}/{split}/**/*.csv",root_dir=_PATH_DATA)
    object_path = glob(f"{folder_name}/{split}/**/object*.tif",root_dir=_PATH_DATA)
    label_path.sort()
    img_path.sort()
    centroid_path.sort()
    object_path.sort()
    
    df = pd.DataFrame()
    
    df["img_path"] = img_path
    df["label_path"] = ""
    df["centroid_path"] = ""
    df["object_path"] = ""
    
    
    
    df["index"] = df.img_path.apply(lambda x: x.split("/")[-1][4:-4])
    df.set_index("index",inplace=True)
    
    for path in label_path:
        df.loc[path.split("/")[-1][6:-4],"label_path"] = path
        
    for path in centroid_path:
        df.loc[path.split("/")[-1][10:-4],"centroid_path"] = path

    for path in object_path:
        df.loc[path.split("/")[-1][8:-4],"object_path"] = path
        
    if gan:
        gan_img_path = glob(f"synthetic_gan/{split}/**/mix*.tif",root_dir=_PATH_DATA)
        gan_img_path.sort()
        df["gan_img_path"] = ""    
        for path in gan_img_path:
            df.loc[path.split("/")[-1][4:-4],"gan_img_path"] = path

    
    df.to_csv(_PATH_DATA+f"/{split}_{version}.csv", index=False, encoding="utf-8")


def create_mixed_and_label_paths_csv():
    img_path = glob("cyclegan_256/*B/**/*.tif",root_dir=_PATH_DATA)
    label_path = glob("metric_distance_labels/real/Mix*/*.tif",root_dir=_PATH_DATA)
    label_path.sort()
    img_path.sort()
    
    df = pd.DataFrame()
    df["img_path"] = img_path
    df["label_path"] = ""
        
    df["index"] = df.img_path.apply(lambda x: x.split("/")[-1][4:-4])
    df.set_index("index",inplace=True)
    
    for path in label_path:
        df.loc[path.split("/")[-1][4:-14],"label_path"] = path
        
    to_be_dropped = []
    for row in df.iterrows():
        if row[1].label_path == "":
            to_be_dropped.append(row[1].name)
    df.drop(to_be_dropped,inplace=True)
    df.to_csv(_PATH_DATA+"/mixed_and_label_paths.csv",index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create synthetic mixes"
    )
    parser.add_argument("--version", type=str, default="kaggle", help="")
    parser.add_argument("--gan", action="store_true", help="")
    parser.add_argument("--train", action="store_true", help="")
    parser.add_argument("--test", action="store_true", help="")
    parser.add_argument("--validation", action="store_true", help="")
    args = parser.parse_args()
    
    if args.test:
        create_csv("test", args.version, args.gan)
    if args.train:
        create_csv("train", args.version, args.gan)
    if args.validation:
        create_csv("validation", args.version, args.gan)
    # create_mixed_and_label_paths_csv()