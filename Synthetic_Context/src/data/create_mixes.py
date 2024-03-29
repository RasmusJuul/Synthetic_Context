import pandas as pd
import tifffile
from py3dbp import Packer, Bin, Item
import re
import numpy as np
from src.data.make_dataset import Label
from src import _PROJECT_ROOT, _PATH_DATA
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from glob import glob
import scipy.ndimage as ndi
from scipy.special import softmax
import argparse

rng = np.random.RandomState(seed=199742)
labels=["BF","CF","PP","MA","BL","ML","SL","WO","BC","GH","AC","GP"]

def get_bin(df, composition, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top):
    packer = Packer()
    packer.add_bin(Bin(f"mix", ax0_top-ax0_bottom, ax1_top-ax1_bottom, ax2_top-ax2_bottom, 999))
    
    for label in composition:
        df_sample = df[df["label"]==label].sample(n=1,random_state=rng)
        shape = [int(s) for s in re.findall(r"\d+", df_sample["size"].item())]
        packer.add_item(Item(df_sample.filename.item(), shape[0], shape[1], shape[2], 1))

    packer.pack(bigger_first=bool(np.random.randint(2)))

    bin = next(iter(packer.bins))

    return bin
    
def find_new_starting_position(start_position, temp_img_shape, temp_img_label, new_mix_label, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top):
    shift_0 = int(start_position[0])
    working_shift_0 = shift_0
    shift_1 = int(start_position[1])
    working_shift_1 = shift_1
    shift_2 = int(start_position[2])
    working_shift_2 = shift_2
    
    
    if shift_0 != 0:
        i = 2
        while True:
            shift_0 = working_shift_0 - int(shift_0/i)
                
            if np.any(new_mix_label[
                      ax0_bottom+shift_0 : ax0_bottom+shift_0 + temp_img_shape[0],
                      ax1_bottom+shift_1 : ax1_bottom+shift_1 + temp_img_shape[1],
                      ax2_bottom+shift_2 : ax2_bottom+shift_2 + temp_img_shape[2],
                      ][temp_img_label != 0] != 0):
                shift_0 = working_shift_0
                i = i*2
                if shift_0/i < 0.1:
                    break
            else:
                if shift_0 >= working_shift_0:
                    break
                elif shift_0 == 0:
                    working_shift_0 = shift_0
                    break
                working_shift_0 = shift_0

    if shift_1 != 0:
        i = 2
        while True:
            shift_1 = working_shift_1 - int(shift_1/i)
                
            if np.any(new_mix_label[
                      ax0_bottom+shift_0 : ax0_bottom+shift_0 + temp_img_shape[0],
                      ax1_bottom+shift_1 : ax1_bottom+shift_1 + temp_img_shape[1],
                      ax2_bottom+shift_2 : ax2_bottom+shift_2 + temp_img_shape[2],
                      ][temp_img_label != 0] != 0):
                shift_1 = working_shift_1
                i = i*2
                if shift_1/i < 0.1:
                    break
            else:
                if shift_1 >= working_shift_1:
                    break
                elif shift_1 == 0:
                    working_shift_1 = shift_1
                    break
                working_shift_1 = shift_1

    if shift_2 != 0:
        i = 2
        while True:
            shift_2 = working_shift_2 - int(shift_2/i)
            
            if np.any(new_mix_label[
                      ax0_bottom+shift_0 : ax0_bottom+shift_0 + temp_img_shape[0],
                      ax1_bottom+shift_1 : ax1_bottom+shift_1 + temp_img_shape[1],
                      ax2_bottom+shift_2 : ax2_bottom+shift_2 + temp_img_shape[2],
                      ][temp_img_label != 0] != 0):
                shift_2 = working_shift_2
                i = i*2
                if shift_2/i < 0.1:
                    break
                    
            else:
                if shift_2 >= working_shift_2:
                    break
                elif shift_2 == 0:
                    working_shift_2 = shift_2
                    break
                working_shift_2 = shift_2
                
    return [working_shift_0,working_shift_1,working_shift_2]

def create_mix(bin, i, mode, new_mix, new_mix_label, df, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top):
    skipped = 0
    all_skipped = False
    for item in bin.items:
        temp_img = tifffile.imread(
                "/".join([_PATH_DATA, "bugnist_256_cut", item.name])
            )
        if item.rotation_type == 1:
            temp_img = np.swapaxes(temp_img, 0, 1)
        elif item.rotation_type == 2:
            temp_img = np.rollaxis(temp_img, 0, 3)
        elif item.rotation_type == 3:
            temp_img = np.swapaxes(temp_img, 0, 2)
        elif item.rotation_type == 4:
            temp_img = np.rollaxis(temp_img, 2, 0)
        elif item.rotation_type == 5:
            temp_img = np.swapaxes(temp_img, 1, 2)

        temp_img_shape = temp_img.shape
        start_position = item.position

        temp_img_label = np.zeros(temp_img_shape, dtype="uint8")
        temp_img_label[temp_img >= 100] = Label.from_abbreviation(
            item.name.split("/")[0]
        ).value

        if np.any(new_mix_label[
                      ax0_bottom+int(start_position[0]) : ax0_bottom+int(start_position[0]) + temp_img_shape[0],
                      ax1_bottom+int(start_position[1]) : ax1_bottom+int(start_position[1]) + temp_img_shape[1],
                      ax2_bottom+int(start_position[2]) : ax2_bottom+int(start_position[2]) + temp_img_shape[2],
                      ][temp_img_label != 0] != 0):
            skipped += 1
            continue
        
        [working_shift_0,working_shift_1,working_shift_2] = find_new_starting_position(start_position, temp_img_shape, temp_img_label, new_mix_label, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top)

        new_mix[
        ax0_bottom+working_shift_0 : ax0_bottom+working_shift_0 + temp_img_shape[0],
        ax1_bottom+working_shift_1 : ax1_bottom+working_shift_1 + temp_img_shape[1],
        ax2_bottom+working_shift_2 : ax2_bottom+working_shift_2 + temp_img_shape[2],
        ] = np.maximum(temp_img,new_mix[
        ax0_bottom+working_shift_0 : ax0_bottom+working_shift_0 + temp_img_shape[0],
        ax1_bottom+working_shift_1 : ax1_bottom+working_shift_1 + temp_img_shape[1],
        ax2_bottom+working_shift_2 : ax2_bottom+working_shift_2 + temp_img_shape[2],
        ])

        
        new_mix_label[
        ax0_bottom+working_shift_0 : ax0_bottom+working_shift_0 + temp_img_shape[0],
        ax1_bottom+working_shift_1 : ax1_bottom+working_shift_1 + temp_img_shape[1],
        ax2_bottom+working_shift_2 : ax2_bottom+working_shift_2 + temp_img_shape[2],
        ] = np.maximum(temp_img_label,new_mix_label[
        ax0_bottom+working_shift_0 : ax0_bottom+working_shift_0 + temp_img_shape[0],
        ax1_bottom+working_shift_1 : ax1_bottom+working_shift_1 + temp_img_shape[1],
        ax2_bottom+working_shift_2 : ax2_bottom+working_shift_2 + temp_img_shape[2],
        ])


        estimated_centroid = ndi.center_of_mass(temp_img,
                                                temp_img_label,
                                                Label.from_abbreviation(
                                                    item.name.split("/")[0]).value)
        
        df.loc[len(df)] = [item.name.split("/")[0],
                           ax2_bottom+working_shift_2+estimated_centroid[2],
                           ax1_bottom+working_shift_1+estimated_centroid[1],
                           ax0_bottom+working_shift_0+estimated_centroid[0]]

    if skipped == len(bin.items):
        all_skipped = True

    return new_mix, new_mix_label, df, all_skipped
    
def generate_mix(df, noise, i, mode, L, with_noise=True):
    df = df[df[mode] == True].reset_index(drop=True)

    if with_noise:
        new_mix = tifffile.imread(rng.choice(noise))
    else:
        new_mix = np.zeros((256, 128, 128), dtype="uint8")
    
    
    ax2_top = new_mix.shape[2]
    ax2_bottom = 0
    ax1_top = new_mix.shape[1]
    ax1_bottom = 0
    ax0_top = new_mix.shape[0]
    ax0_bottom = 0

    if with_noise:
        for i in range(new_mix.shape[2]):
            if np.any(new_mix[:, :, i] != 0):
                ax2_bottom = i
                break
        for i in range(new_mix.shape[2] - 1, -1, -1):
            if np.any(new_mix[:, :, i] != 0):
                ax2_top = i
                break
        for i in range(new_mix.shape[1]):
            if np.any(new_mix[:, i, :] != 0):
                ax1_bottom = i
                break
        for i in range(new_mix.shape[1] - 1, -1, -1):
            if np.any(new_mix[:, i, :] != 0):
                ax1_top = i
                break
        for i in range(new_mix.shape[0]):
            if np.any(new_mix[i, :, :] != 0):
                ax0_bottom = i
                break
        for i in range(new_mix.shape[0] - 1, -1, -1):
            if np.any(new_mix[i, :, :] != 0):
                ax0_top = i
                break

    new_mix_label = np.zeros((256, 128, 128), dtype="uint8")
    df_new = pd.DataFrame(columns=["Caption","PosX","PosY","PosZ"])
    
    if len(L) != 0:
        prob=[]
        for name in labels:
            prob.append(1-(L.count(name)*50)/len(L))
        prob = softmax(prob)
    else:
        prob=[1/12]*12
        
    while True:
        composition = rng.choice(labels,size=20,p=prob)
        bin = get_bin(df, composition, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top)
    
        new_mix, new_mix_label, df_new, all_skipped = create_mix(bin, i, mode, new_mix, new_mix_label, df_new, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top)
        if all_skipped:
            break
            
    L.extend(df_new.Caption.tolist())
    return L, new_mix, new_mix_label, df_new

def save_files(mode,new_mix,new_mix_label,df,i,folder_name):
    subfolders = glob(
        "**/", root_dir=f"{_PATH_DATA}/{folder_name}/{mode}"
    )
    if len(subfolders) == 0:
        os.makedirs(
            f"{_PATH_DATA}/{folder_name}/{mode}/{str(len(subfolders)).zfill(3)}",
            exist_ok=True,
        )
        subfolder = str(len(subfolders)).zfill(3)
    else:
        for subfolder in subfolders:
            if (
                len(
                    glob(
                        subfolder + "*.tif",
                        root_dir=f"{_PATH_DATA}/{folder_name}/{mode}")
                )
                < 200
            ):
                break

    if (
        len(
            glob(
                subfolder + "*.tif",
                root_dir=f"{_PATH_DATA}/{folder_name}/{mode}"),
        )
        >= 200
    ):
        os.makedirs(
            f"{_PATH_DATA}/{folder_name}/{mode}/{str(len(subfolders)).zfill(3)}",
            exist_ok=True,
        )
        subfolder = str(len(subfolders)).zfill(3)

    tifffile.imwrite(
        f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/mix_{str(i).zfill(5)}.tif",
        new_mix,
    )
    tifffile.imwrite(
        f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/label_{str(i).zfill(5)}.tif",
        new_mix_label,
    )

    df.to_csv(f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/centroids_{str(i).zfill(5)}.csv",sep=";",index=False,encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create synthetic mixes"
    )
    parser.add_argument("--no_noise", action="store_true", help="if true don't insert insects into noise (leafs, sawdust, cotton)")
    args = parser.parse_args()

    df = pd.read_csv("/".join([_PATH_DATA, "bugnist_256_cut/splits.csv"]))
    df["label"] = df.filename.apply(lambda x: x.split("/")[0])
    noise = glob(_PATH_DATA + "/noise/*.tif")
    
    with_noise = not args.no_noise
    
    if with_noise:
        folder_name = "synthetic_mixed_256_v3"
    else:
        folder_name = "synthetic_mixed_256_v2"

    os.makedirs(f"{_PATH_DATA}/{folder_name}/train", exist_ok=True)
    os.makedirs(f"{_PATH_DATA}/{folder_name}/test", exist_ok=True)
    os.makedirs(f"{_PATH_DATA}/{folder_name}/validation", exist_ok=True)

    # L = []
    # for i in tqdm(range(50000), unit="image", desc="creating mixed images"):
    #     L, new_mix, new_mix_label, df_new = generate_mix(df,noise,i,"train",L,with_noise)
    #     save_files("train",new_mix,new_mix_label,df_new,i,folder_name)
    L = []
    for i in tqdm(range(500), unit="image", desc="creating mixed images"):
        L, new_mix, new_mix_label, df_new = generate_mix(df,noise,i,"test",L,with_noise)
        save_files("test",new_mix,new_mix_label,df_new,i,folder_name)
    L = []
    for i in tqdm(range(500), unit="image", desc="creating mixed images"):
        L, new_mix, new_mix_label, df_new = generate_mix(df,noise,i,"validation",L,with_noise)
        save_files("validation",new_mix,new_mix_label,df_new,i,folder_name)

