from glob import glob
import pandas as pd
import numpy as np
from src import _PATH_DATA
import argparse


def create_csv(split,no_noise=False,old=False,gan=False):
    if no_noise:
        folder_name = "synthetic_mixed_256_v2"
    elif old:
        folder_name = "synthetic_mixed_256_v1"
    else:
        folder_name = "synthetic_mixed_256_v3"
        
    img_path = glob(f"{folder_name}/{split}/**/mix*.tif",root_dir=_PATH_DATA)
    label_path = glob(f"{folder_name}/{split}/**/label*.tif",root_dir=_PATH_DATA)
    centroid_path = glob(f"{folder_name}/{split}/**/*.csv",root_dir=_PATH_DATA)
    label_path.sort()
    img_path.sort()
    centroid_path.sort()
    
    df = pd.DataFrame()
    
    df["img_path"] = img_path
    df["label_path"] = ""
    df["centroid_path"] = ""
    
    
    
    df["index"] = df.img_path.apply(lambda x: x.split("/")[-1][4:-4])
    df.set_index("index",inplace=True)
    
    for path in label_path:
        df.loc[path.split("/")[-1][6:-4],"label_path"] = path
        
    for path in centroid_path:
        df.loc[path.split("/")[-1][10:-4],"centroid_path"] = path
        
    if gan:
        gan_img_path = glob(f"synthetic_gan/{split}/**/mix*.tif",root_dir=_PATH_DATA)
        gan_img_path.sort()
        df["gan_img_path"] = ""    
        for path in gan_img_path:
            df.loc[path.split("/")[-1][4:-4],"gan_img_path"] = path

    if no_noise:
        df.to_csv(_PATH_DATA+f"/{split}_v2.csv", index=False, encoding="utf-8")
    elif old:
        df.to_csv(_PATH_DATA+f"/{split}_v1.csv", index=False, encoding="utf-8")
    else:
        df.to_csv(_PATH_DATA+f"/{split}_v3.csv", index=False, encoding="utf-8")

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
    parser.add_argument("--no_noise", action="store_true", help="")
    parser.add_argument("--old", action="store_true", help="")
    parser.add_argument("--gan", action="store_true", help="")
    parser.add_argument("--train", action="store_true", help="")
    parser.add_argument("--test", action="store_true", help="")
    parser.add_argument("--validation", action="store_true", help="")
    args = parser.parse_args()
    
    if args.test:
        create_csv("test",args.no_noise,args.old,args.gan)
    if args.train:
        create_csv("train",args.no_noise,args.old,args.gan)
    if args.validation:
        create_csv("validation",args.no_noise,args.old,args.gan)
    # create_mixed_and_label_paths_csv()