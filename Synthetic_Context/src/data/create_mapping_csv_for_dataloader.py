from glob import glob
import pandas as pd
import numpy as np
from src import _PATH_DATA


def create_csv(split):
    img_path = glob(f"synthetic_mixed_256/{split}/**/mix*.tif",root_dir=_PATH_DATA)
    gan_img_path = glob(f"synthetic_gan/{split}/**/mix*.tif",root_dir=_PATH_DATA)
    label_path = glob(f"synthetic_mixed_256/{split}/**/label*.tif",root_dir=_PATH_DATA)
    centroid_path = glob(f"synthetic_mixed_256/{split}/**/*.csv",root_dir=_PATH_DATA)
    label_path.sort()
    img_path.sort()
    centroid_path.sort()
    gan_img_path.sort()
    df = pd.DataFrame()
    
    df["img_path"] = img_path
    df["label_path"] = ""
    df["centroid_path"] = ""
    df["gan_img_path"] = ""
    
    df["index"] = df.img_path.apply(lambda x: x.split("/")[-1][4:-4])
    df.set_index("index",inplace=True)
    
    for path in label_path:
        df.loc[path.split("/")[-1][6:-4],"label_path"] = path
        
    for path in centroid_path:
        df.loc[path.split("/")[-1][10:-4],"centroid_path"] = path
        
    for path in gan_img_path:
        df.loc[path.split("/")[-1][4:-4],"gan_img_path"] = path
    
    df.to_csv(_PATH_DATA+f"/{split}.csv", index=False)

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
    create_csv("test")
    create_csv("train")
    create_csv("validation")
    create_mixed_and_label_paths_csv()