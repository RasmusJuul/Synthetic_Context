import pandas as pd
import tifffile
from py3dbp import Packer, Bin, Item
import re
import numpy as np
from src.data.dataloaders import Label
from src import _PROJECT_ROOT, _PATH_DATA
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from glob import glob
import scipy.ndimage as ndi
from scipy.special import softmax
import argparse


labels=["BF","CF","PP","MA","BL","ML","SL","WO","BC","GH","AC","BP"]

def get_bin(df, composition, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top,rng):
    packer = Packer()
    packer.add_bin(Bin(f"mix", ax0_top-ax0_bottom, ax1_top-ax1_bottom, ax2_top-ax2_bottom, 999))
    
    for label in composition:
        df_sample = df[df["label"]==label].sample(n=1,random_state=rng)
        shape = [int(s) for s in re.findall(r"\d+", df_sample["size"].item())]
        packer.add_item(Item(df_sample.file_name.item(), shape[0], shape[1], shape[2], 1))

    packer.pack(bigger_first=bool(rng.randint(2)))

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

def create_mix(bin, i, mode, new_mix, new_mix_label, new_mix_objects, df, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top, objects_added):
    skipped = 0
    all_skipped = False
    for item in bin.items:
        temp_img = tifffile.imread(
                "/".join([_PATH_DATA, item.name])
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

        # boundary = np.zeros(temp_img_shape, dtype="uint8")
        # boundary[temp_img >= 50] = 1
        # boundary = ndi.binary_dilation(boundary,iterations=2) - boundary
        
        temp_img_label = np.zeros(temp_img_shape, dtype="uint8")
        temp_img_label[temp_img >= 50] = Label.from_abbreviation(
            item.name.split("/")[-2]
        ).value
        # temp_img_label[boundary == 1] = Label.from_abbreviation(
        #     item.name.split("/")[-2]
        # ).value+12

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

        objects_added += 1
        
        new_mix_objects[
        ax0_bottom+working_shift_0 : ax0_bottom+working_shift_0 + temp_img_shape[0],
        ax1_bottom+working_shift_1 : ax1_bottom+working_shift_1 + temp_img_shape[1],
        ax2_bottom+working_shift_2 : ax2_bottom+working_shift_2 + temp_img_shape[2],
        ][temp_img_label != 0] = objects_added

        estimated_centroid = ndi.center_of_mass(temp_img,
                                                temp_img_label,
                                                Label.from_abbreviation(
                                                    item.name.split("/")[-2]).value)
        
        df.loc[len(df)] = [item.name.split("/")[-2],
                           ax2_bottom+working_shift_2+estimated_centroid[2],
                           ax1_bottom+working_shift_1+estimated_centroid[1],
                           ax0_bottom+working_shift_0+estimated_centroid[0]]

    if skipped == len(bin.items):
        all_skipped = True

    return new_mix, new_mix_label, new_mix_objects, df, all_skipped, objects_added
    

def generate_mix(df, noise, i, mode, kaggle=False, with_noise=True, size=(256,128,128),folder_name="test",count=1,seed=42):
    df = df[df[mode] == True].reset_index(drop=True)
    rng = np.random.RandomState(seed=seed)
    i_counter = i
    if with_noise:
        new_mix = tifffile.imread(rng.choice(noise))
        new_mix = ndi.rotate(new_mix, rng.randint(0, 360), axes=(1, 0), reshape=False)
        new_mix = ndi.rotate(new_mix, rng.randint(0, 360), axes=(1, 2), reshape=False)
        new_mix = ndi.rotate(new_mix, rng.randint(0, 360), axes=(0, 2), reshape=False)
    else:
        new_mix = np.zeros(size, dtype="uint8")
    
    
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

    new_mix_label = np.zeros(size, dtype="uint8")
    new_mix_objects = np.zeros(size, dtype="uint8")
    df_new = pd.DataFrame(columns=["Caption","PosX","PosY","PosZ"])

    objects_added = 0

    prob=[1/12]*12
        
    while True:
        composition = rng.choice(labels,size=20,p=prob)
        bin = get_bin(df, composition, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top,rng)
    
        new_mix, new_mix_label, new_mix_objects, df_new, all_skipped, objects_added = create_mix(bin, i, mode, new_mix, new_mix_label, new_mix_objects, df_new, ax0_bottom, ax0_top, ax1_bottom, ax1_top, ax2_bottom, ax2_top, objects_added)
        if all_skipped:
            break
            
    save_files(mode,new_mix,new_mix_label,new_mix_objects,df_new,i_counter,folder_name,count)

def save_files(mode,new_mix,new_mix_label,new_mix_objects,df,i,folder_name,count):
    num_subfolders = int(count/100)
    subfolder = str(i%num_subfolders).zfill(3)

    
    tifffile.imwrite(
        f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/mix_{str(i).zfill(5)}.tif",
        new_mix,
    )
    tifffile.imwrite(
        f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/label_{str(i).zfill(5)}.tif",
        new_mix_label,
    )
    tifffile.imwrite(
        f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/objects_{str(i).zfill(5)}.tif",
        new_mix_objects,
    )

    df.to_csv(f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/centroids_{str(i).zfill(5)}.csv",sep=";",index=False,encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create synthetic mixes"
    )
    parser.add_argument("--no-noise", action="store_true", help="if true don't insert insects into noise (leafs, sawdust, cotton)")
    parser.add_argument("--kaggle", action="store_true", help="kaggle restrictions")
    args = parser.parse_args()

    df = pd.read_csv("/".join([_PATH_DATA, "BugNIST_DATA/cut_sizes.csv"]))
    # df["label"] = df.filename.apply(lambda x: x.split("/")[0])
    noise = glob(_PATH_DATA + "/noise2/*.tif")
    
    with_noise = not args.no_noise
    
    if with_noise:
        if args.kaggle:
            folder_name = "synthetic_mixed_kaggle_noisy"
        else:
            folder_name = "synthetic_mixed_256_v3"
    else:
        if args.kaggle:
            folder_name = "synthetic_mixed_kaggle"
        else:
            folder_name = "synthetic_mixed_256_v2"

    os.makedirs(f"{_PATH_DATA}/{folder_name}/train", exist_ok=True)
    os.makedirs(f"{_PATH_DATA}/{folder_name}/test", exist_ok=True)
    os.makedirs(f"{_PATH_DATA}/{folder_name}/validation", exist_ok=True)


    count_train = 25000
    count_test = 500
    count_validation = 500

    for i in range(int(count_train / 100)):
        os.makedirs(
            _PATH_DATA + f"/{folder_name}/train/{str(i).zfill(3)}", exist_ok=True
        )
        
    for i in range(int(count_test / 100)):
        os.makedirs(
            _PATH_DATA + f"/{folder_name}/test/{str(i).zfill(3)}", exist_ok=True
        )
    for i in range(int(count_validation / 100)):
        os.makedirs(
            _PATH_DATA + f"/{folder_name}/validation/{str(i).zfill(3)}", exist_ok=True
        )

    
    rng_train = np.random.SeedSequence(42).generate_state(count_train)
    rng_test = np.random.SeedSequence(1337).generate_state(count_test)
    rng_validation = np.random.SeedSequence(1997).generate_state(count_validation)
    
    Parallel(n_jobs=-1)(delayed(generate_mix)
                        (df,noise,i,"train",kaggle=args.kaggle,with_noise=with_noise,size=(128,92,92),folder_name=folder_name,count=count_train,seed=rng_train[i])
                        for i in tqdm(range(count_train), unit="image", desc="creating mixed images"))
    Parallel(n_jobs=-1)(delayed(generate_mix)
                        (df,noise,i,"test",kaggle=args.kaggle,with_noise=with_noise,size=(128,92,92),folder_name=folder_name,count=count_test,seed=rng_test[i])
                        for i in tqdm(range(count_test), unit="image", desc="creating mixed images"))
    Parallel(n_jobs=-1)(delayed(generate_mix)
                        (df,noise,i,"validation",kaggle=args.kaggle,with_noise=with_noise,size=(128,92,92),folder_name=folder_name,count=count_validation,seed=rng_validation[i])
                        for i in tqdm(range(count_validation), unit="image", desc="creating mixed images"))
        
    # L = []
    # for i in tqdm(range(25000), unit="image", desc="creating mixed images"):
    #     L, new_mix, new_mix_label, df_new, new_mix_objects = generate_mix(df,noise,i,"train",L,kaggle=args.kaggle,with_noise=with_noise,size=(128,92,92))
    #     save_files("train",new_mix,new_mix_label,new_mix_objects,df_new,i,folder_name)
    # L = []
    # for i in tqdm(range(500), unit="image", desc="creating mixed images"):
    #     L, new_mix, new_mix_label, df_new, new_mix_objects = generate_mix(df,noise,i,"test",L,kaggle=args.kaggle,with_noise=with_noise,size=(128,92,92))
    #     save_files("test",new_mix,new_mix_label,new_mix_objects,df_new,i,folder_name)
    # L = []
    # for i in tqdm(range(500), unit="image", desc="creating mixed images"):
    #     L, new_mix, new_mix_label, df_new, new_mix_objects = generate_mix(df,noise,i,"validation",L,kaggle=args.kaggle,with_noise=with_noise,size=(128,92,92))
    #     save_files("validation",new_mix,new_mix_label,new_mix_objects,df_new,i,folder_name)

