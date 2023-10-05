from glob import glob
import pandas as pd
import numpy as np
import scipy.ndimage as ndi
import tifffile
from src import _PATH_DATA
from src.data.dataloaders import Label
from skimage.morphology import skeletonize_3d
from joblib import Parallel, delayed
import os
from tqdm import tqdm

def make_labels(idx,df):
    df_label = pd.read_csv("/".join([_PATH_DATA,df.label_path.iloc[idx]]),sep=";")
    df_label = df_label.round().astype({"PosX":"int16","PosY":"int16","PosZ":"int16"})
    
    df_label["PosX_resized"] = (df_label.PosX/650)*128
    df_label["PosY_resized"] = (df_label.PosY/650)*128
    df_label["PosZ_resized"] = (df_label.PosZ/900)*256
    
    df_label = df_label.round().astype({"PosX_resized":"int16","PosY_resized":"int16","PosZ_resized":"int16"})
    img = tifffile.imread("/".join([_PATH_DATA,df.img_256_path.iloc[idx]]))

    img2 = img.copy()
    for i in range(10):
        img2 = ndi.grey_closing(img2,structure=ndi.generate_binary_structure(3,2))
    for i in range(3):
        img2 = ndi.grey_erosion(img2,structure=ndi.generate_binary_structure(3,2))
    for i in range(6):
        img2 = ndi.grey_opening(img2,structure=ndi.generate_binary_structure(3,2))
    
    img2[img2 < 25] = 0
    skeleton = skeletonize_3d(img2)
    
    markers = np.zeros(img.shape,dtype="int8")
    markers[skeleton < 75] = -1
    markers[skeleton >= 75] = 0
    
    for item in df_label.iterrows():
        for i in range(-5,6):
            for j in range(-5,6):
                for k in range(-5,6):
                    markers[item[1].PosZ_resized+i,item[1].PosY_resized+k,item[1].PosX_resized+j] = Label.from_abbreviation(item[1].Caption).value
    
    skeleton_label = ndi.watershed_ift(skeleton,markers,structure=ndi.generate_binary_structure(3,3))
    skeleton_label[skeleton_label == -1] = 0

    for item in df_label.iterrows():
        for i in range(-5,6):
            for j in range(-5,6):
                for k in range(-5,6):
                    skeleton_label[item[1].PosZ_resized+i,item[1].PosY_resized+k,item[1].PosX_resized+j] = Label.from_abbreviation(item[1].Caption).value
    
    
    img2 = np.array(img2).astype(np.float32)
    mn = np.min(img2)
    mx = np.max(img2)
    norm = (img2 - mn) * (1.0 / (mx - mn))
    img2 = (norm*255).astype("uint8")
    
    skeleton_label[img2 < 15] = -1
    label_img = ndi.watershed_ift(img, skeleton_label,structure=ndi.generate_binary_structure(3,2))
    label_img[label_img == -1] = 0
    
    for item in df_label.iterrows():
        for i in range(-5,6):
            for j in range(-5,6):
                for k in range(-5,6):
                    label_img[item[1].PosZ_resized+i,item[1].PosY_resized+k,item[1].PosX_resized+j] = Label.from_abbreviation(item[1].Caption).value
    
    os.makedirs(
            f"{_PATH_DATA}/metric_labels/{df.iloc[idx].name[:-4]}",
            exist_ok=True,
        )
    tifffile.imwrite(f"{_PATH_DATA}/metric_labels/{df.iloc[idx].name[:-4]}/{df.iloc[idx].name}_label.tif",label_img.astype("uint8"))



if __name__ == "__main__":
    img_path = glob("mixed_crop/Mix*/*.tif",root_dir=_PATH_DATA)
    label_path = glob("mixed_crop/**/*.csv",root_dir=_PATH_DATA)
    label_path.sort()
    img_path.sort()
    
    df = pd.DataFrame(columns=["img_path","label_path"])
    last = 0
    for i in range(len(label_path)):
        label = label_path[i].split("/")[-1]
        for j in range(last,len(img_path)):
            img = img_path[j].split("/")[-1]
            if img[:-4] == label[:-14]:
                df.loc[len(df)] = [img_path[j],label_path[i]]
                last = j
    
    df["index"] = df.img_path.apply(lambda x: x.split("/")[-1][:-4])
    df.set_index("index",inplace=True)
    df["img_256_path"] = ""
    
    img_256_path = glob("cyclegan_256/*B/**/*.tif",root_dir=_PATH_DATA)
    for path in img_256_path:
        df.loc[path.split("/")[-1][:-4],"img_256_path"] = path
    df.dropna(inplace=True)

    Parallel(n_jobs=-1)(delayed(make_labels)(i, df) for i in tqdm(range(len(df)), unit="image", desc="creating labels"))