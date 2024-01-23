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

def get_bin(df):
    packer = Packer()
    packer.add_bin(Bin(f'mix', 256, 128, 128, 999))
    img_idx = np.random.choice(len(df),15,replace=False)
    
    for idx in img_idx:
        item = df.loc[idx]
        shape = [int(s) for s in re.findall(r'\d+',item['size'])]
        packer.add_item(Item(item.filename, shape[0], shape[1], shape[2], 1))
    
    packer.pack(bigger_first=bool(np.random.randint(2)))

    bin = next(iter(packer.bins))

    return bin

def create_mix(bin,i):
    new_mix = np.zeros((256,128,128),dtype='uint8')
    new_mix_label = np.zeros((256,128,128),dtype='uint8')
    for item in bin.items:
        temp_img = tifffile.imread("/".join([_PATH_DATA, "bugnist_256_cut", item.name]))
        if item.rotation_type == 1:
            temp_img = np.swapaxes(temp_img,0,1)
        elif item.rotation_type == 2:
            temp_img = np.rollaxis(temp_img,0,3)
        elif item.rotation_type == 3:
            temp_img = np.swapaxes(temp_img,0,2)
        elif item.rotation_type == 4:
            temp_img = np.rollaxis(temp_img,2,0)
        elif item.rotation_type == 5:
            temp_img = np.swapaxes(temp_img,1,2)
        
        temp_img_shape = temp_img.shape
        start_position = item.position
        temp_img_label = np.zeros(temp_img_shape,dtype='uint8')
        temp_img_label[temp_img >= 100] = Label.from_abbreviation(item.name.split("/")[0]).value
        
        
        new_mix[int(start_position[0]):int(start_position[0])+temp_img_shape[0],
                int(start_position[1]):int(start_position[1])+temp_img_shape[1],
                int(start_position[2]):int(start_position[2])+temp_img_shape[2]] = temp_img

        new_mix_label[int(start_position[0]):int(start_position[0])+temp_img_shape[0],
                int(start_position[1]):int(start_position[1])+temp_img_shape[1],
                int(start_position[2]):int(start_position[2])+temp_img_shape[2]] = temp_img_label

    return new_mix, new_mix_label
    

def save_mix(new_mix, new_mix_label, mode, i, count):
    folder_name = "synthetic_mixed_256_v1"
    num_subfolders = int(count*2/200)
    subfolder = str(i%num_subfolders).zfill(3)

    tifffile.imwrite(f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/mix_{str(i).zfill(5)}.tif", new_mix)
    tifffile.imwrite(f"{_PATH_DATA}/{folder_name}/{mode}/{subfolder}/label_{str(i).zfill(5)}.tif", new_mix_label)

def generate_mix(df, mode, i, count):
    df = df[df[mode] == True].reset_index(drop=True)
    
    bin = get_bin(df)
    new_mix, new_mix_label = create_mix(bin, i)
    save_mix(new_mix, new_mix_label, mode, i, count)


if __name__=="__main__":

    df = pd.read_csv("/".join([_PATH_DATA,"bugnist_256_cut/splits.csv"]))
    df["label"] = df.filename.apply(lambda x: x.split("/")[0])

    count_train = 50000
    count_test = 500
    count_validation = 500
    folder_name = "synthetic_mixed_256_v1"
    
    for i in range(int(count_train*2 / 200)):
        os.makedirs(
            _PATH_DATA + f"/{folder_name}/train/{str(i).zfill(3)}", exist_ok=True
        )
        
    for i in range(int(count_test*2 / 200)):
        os.makedirs(
            _PATH_DATA + f"/{folder_name}/test/{str(i).zfill(3)}", exist_ok=True
        )
    for i in range(int(count_validation*2 / 200)):
        os.makedirs(
            _PATH_DATA + f"/{folder_name}/validation/{str(i).zfill(3)}", exist_ok=True
        )
    
    Parallel(n_jobs=-1)(delayed(generate_mix)(df,"train", i,count_train) for i in tqdm(range(count_train), unit="image", desc="creating mixed images"))
    Parallel(n_jobs=-1)(delayed(generate_mix)(df,"validation", i, count_validation) for i in tqdm(range(count_validation), unit="image", desc="creating mixed images"))
    Parallel(n_jobs=-1)(delayed(generate_mix)(df,"test", i,count_test) for i in tqdm(range(count_test), unit="image", desc="creating mixed images"))