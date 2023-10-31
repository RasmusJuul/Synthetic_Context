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
import argparse


def make_labels(idx,df,synthetic):
    df_label = pd.read_csv("/".join([_PATH_DATA,df.label_path.iloc[idx]]),sep=";")
    df_label = df_label.round().astype({"PosX":"int16","PosY":"int16","PosZ":"int16"})
    
    if not synthetic:
        df_label["PosX_resized"] = (df_label.PosX/650)*128
        df_label["PosY_resized"] = (df_label.PosY/650)*128
        df_label["PosZ_resized"] = (df_label.PosZ/900)*256
        df_label = df_label.round().astype({"PosX_resized":"int16","PosY_resized":"int16","PosZ_resized":"int16"})
        img = tifffile.imread("/".join([_PATH_DATA,df.img_256_path.iloc[idx]]))
    else:
        img = tifffile.imread("/".join([_PATH_DATA,df.img_path.iloc[idx]]))

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
        for i in range(-7,8):
            for j in range(-7,8):
                for k in range(-7,8):
                    if synthetic:
                        try:
                            markers[item[1].PosZ+i,item[1].PosY+k,item[1].PosX+j] = Label.from_abbreviation(item[1].Caption).value
                        except:
                            print(f"centroid too close to the edge in {df.iloc[idx].img_path.split('/')[-1][:-4]}")
                    else:
                        markers[item[1].PosZ_resized+i,item[1].PosY_resized+k,item[1].PosX_resized+j] = Label.from_abbreviation(item[1].Caption).value
                        
    skeleton_label = ndi.watershed_ift(skeleton,markers,structure=ndi.generate_binary_structure(3,3))
    skeleton_label[skeleton_label == -1] = 0

    for item in df_label.iterrows():
        for i in range(-7,8):
            for j in range(-7,8):
                for k in range(-7,8):
                    if synthetic:
                        try:
                            skeleton_label[item[1].PosZ+i,item[1].PosY+k,item[1].PosX+j] = Label.from_abbreviation(item[1].Caption).value
                        except:
                            continue
                    else:
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
        for i in range(-7,8):
            for j in range(-7,8):
                for k in range(-7,8):
                    if synthetic:
                        try:
                            label_img[item[1].PosZ+i,item[1].PosY+k,item[1].PosX+j] = Label.from_abbreviation(item[1].Caption).value
                        except:
                            continue
                    else:
                        label_img[item[1].PosZ_resized+i,item[1].PosY_resized+k,item[1].PosX_resized+j] = Label.from_abbreviation(item[1].Caption).value
    
    # os.makedirs(
    #         f"{_PATH_DATA}/metric_labels/{df.iloc[idx].name[:-4]}",
    #         exist_ok=True,
    #     )
    # tifffile.imwrite(f"{_PATH_DATA}/metric_labels/{df.iloc[idx].name[:-4]}/{df.iloc[idx].name}_label.tif",label_img.astype("uint8"))

    label_img = label_img.astype("int16")
    dists = np.zeros((13,256,128,128))
    label_img[label_img == 0] = -1
    for i in range(1,13):
        label_img[label_img == i] = 0
        dists[i,:,:,:] = ndi.distance_transform_cdt(label_img)
        label_img[label_img == 0] = i

    dists[0,:,:,:] = np.inf
    dists[dists == -1] = np.inf
    dists = np.argmin(dists,axis=0)
    
    if synthetic:
        os.makedirs(f"{_PATH_DATA}/metric_distance_labels/synthetic/{df.iloc[idx].img_path.split('/')[-2]}", exist_ok=True)
        tifffile.imwrite(f"{_PATH_DATA}/metric_distance_labels/synthetic/{df.iloc[idx].img_path.split('/')[-2]}/{df.iloc[idx].img_path.split('/')[-1][:-4]}_distances.tif",dists.astype("uint8"))
    else:
        os.makedirs(f"{_PATH_DATA}/metric_distance_labels/real/{df.iloc[idx].img_path.split('/')[-2]}", exist_ok=True)
        tifffile.imwrite(f"{_PATH_DATA}/metric_distance_labels/real/{df.iloc[idx].img_path.split('/')[-2]}/{df.iloc[idx].img_path.split('/')[-1][:-4]}_distances.tif",dists.astype("uint8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create labels from centroids"
    )
    parser.add_argument("--synthetic", action="store_true", help="if true create labels for the synthetic data")
    args = parser.parse_args()

    if args.synthetic:
        img_path = glob("synthetic_mixed_256/test/**/mix*.tif",root_dir=_PATH_DATA)
        label_path = glob("synthetic_mixed_256/test/**/*.csv",root_dir=_PATH_DATA)
        label_path.sort()
        img_path.sort()
        
        df = pd.DataFrame(columns=["img_path","label_path"])
        last = 0
        for i in range(len(label_path)):
            label = label_path[i].split("/")[-1]
            for j in range(last,len(img_path)):
                img = img_path[j].split("/")[-1]
                if img[4:-4] == label[10:-4]:
                    df.loc[len(df)] = [img_path[j],label_path[i]]
                    last = j

        df["index"] = df.img_path.apply(lambda x: x.split("/")[-1][:-4])
        df.set_index("index",inplace=True)
        
    else:
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

    Parallel(n_jobs=-1)(delayed(make_labels)(i, df,args.synthetic) for i in tqdm(range(len(df)), unit="image", desc="creating labels"))