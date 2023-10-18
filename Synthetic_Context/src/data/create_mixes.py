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


def get_bin(df):
    packer = Packer()
    packer.add_bin(Bin(f"mix", 256, 128, 128, 999))
    img_idx = np.random.choice(len(df), 30, replace=False)

    for idx in img_idx:
        item = df.loc[idx]
        shape = [int(s) for s in re.findall(r"\d+", item["size"])]
        packer.add_item(Item(item.filename, shape[0], shape[1], shape[2], 1))

    packer.pack(bigger_first=bool(np.random.randint(2)))

    bin = next(iter(packer.bins))

    return bin


def create_mix(bin, i, mode):
    new_mix = np.zeros((256, 128, 128), dtype="uint8")
    new_mix_label = np.zeros((256, 128, 128), dtype="uint8")
    df = pd.DataFrame(columns=["Caption","PosX","PosY","PosZ"])
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

        

        new_mix[
            int(start_position[0]) : int(start_position[0]) + temp_img_shape[0],
            int(start_position[1]) : int(start_position[1]) + temp_img_shape[1],
            int(start_position[2]) : int(start_position[2]) + temp_img_shape[2],
        ] = temp_img

        
        temp_img_label = np.zeros(temp_img_shape, dtype="uint8")
        temp_img_label[temp_img >= 100] = Label.from_abbreviation(
            item.name.split("/")[0]
        ).value

        new_mix_label[
            int(start_position[0]) : int(start_position[0]) + temp_img_shape[0],
            int(start_position[1]) : int(start_position[1]) + temp_img_shape[1],
            int(start_position[2]) : int(start_position[2]) + temp_img_shape[2],
        ] = temp_img_label


        estimated_centroid = ndi.center_of_mass(temp_img,
                                                temp_img_label,
                                                Label.from_abbreviation(
                                                    item.name.split("/")[0]).value)
        
        df.loc[len(df)] = [item.name.split("/")[0],
                           int(start_position[2])+estimated_centroid[2],
                           int(start_position[1])+estimated_centroid[1],
                           int(start_position[0])+estimated_centroid[0]]

    subfolders = glob(
        "**/", root_dir="/".join([_PATH_DATA, f"synthetic_mixed_256/{mode}"])
    )
    if len(subfolders) == 0:
        os.makedirs(
            _PATH_DATA + f"/synthetic_mixed_256/{mode}/{str(len(subfolders)).zfill(2)}",
            exist_ok=True,
        )
        subfolder = str(len(subfolders)).zfill(2)
    else:
        for subfolder in subfolders:
            if (
                len(
                    glob(
                        subfolder + "*.tif",
                        root_dir="/".join([_PATH_DATA, f"synthetic_mixed_256/{mode}"]),
                    )
                )
                < 250
            ):
                break

    if (
        len(
            glob(
                subfolder + "*.tif",
                root_dir="/".join([_PATH_DATA, f"synthetic_mixed_256/{mode}"]),
            )
        )
        >= 250
    ):
        os.makedirs(
            _PATH_DATA + f"/synthetic_mixed_256/{mode}/{str(len(subfolders)).zfill(2)}",
            exist_ok=True,
        )
        subfolder = str(len(subfolders)).zfill(2)

    tifffile.imwrite(
        f"{_PATH_DATA}/synthetic_mixed_256/{mode}/{subfolder}/mix_{str(i).zfill(5)}.tif",
        new_mix,
    )
    tifffile.imwrite(
        f"{_PATH_DATA}/synthetic_mixed_256/{mode}/{subfolder}/label_{str(i).zfill(5)}.tif",
        new_mix_label,
    )

    df.to_csv(f"{_PATH_DATA}/synthetic_mixed_256/{mode}/{subfolder}/centroids_{str(i).zfill(5)}.csv",sep=";",index=False)


def generate_mix(df, i, mode):
    df = df[df[mode] == True].reset_index(drop=True)

    bin = get_bin(df)
    create_mix(bin, i, mode)


if __name__ == "__main__":
    df = pd.read_csv("/".join([_PATH_DATA, "bugnist_256_cut/splits.csv"]))

    for i in range(int(2500*3 / 250)):
        os.makedirs(
            _PATH_DATA + f"/synthetic_mixed_256/train/{str(i).zfill(2)}", exist_ok=True
        )
    for i in range(int(500*3 / 250)):
        os.makedirs(
            _PATH_DATA + f"/synthetic_mixed_256/test/{str(i).zfill(2)}", exist_ok=True
        )
    for i in range(int(500*3 / 250)):
        os.makedirs(
            _PATH_DATA + f"/synthetic_mixed_256/validation/{str(i).zfill(2)}",
            exist_ok=True,
        )

    Parallel(n_jobs=-1)(
        delayed(generate_mix)(df, i, "train")
        for i in tqdm(range(2500), unit="image", desc="creating mixed images")
    )
    Parallel(n_jobs=-1)(
        delayed(generate_mix)(df, i, "test")
        for i in tqdm(range(500), unit="image", desc="creating mixed images")
    )
    Parallel(n_jobs=-1)(
        delayed(generate_mix)(df, i, "validation")
        for i in tqdm(range(500), unit="image", desc="creating mixed images")
    )
