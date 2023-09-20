import pandas as pd
import tifffile
from py3dbp import Packer, Bin, Item
import re
import numpy as np
from src.data.make_dataset import Label
from src import _PROJECT_ROOT, _PATH_DATA
from joblib import Parallel, delayed
from tqdm import tqdm

def get_bin(df):
    packer = Packer()
    packer.add_bin(Bin(f'mix', 256, 128, 128, 999))
    img_idx = np.random.choice(len(df),20,replace=False)
    
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
        temp_img = tifffile.imread("/".join([_PROJECT_ROOT,item.name]))
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
        temp_img_label[temp_img >= 100] = Label.from_abbreviation(item.name.split("/")[2]).value
        
        
        new_mix[int(start_position[0]):int(start_position[0])+temp_img_shape[0],
                int(start_position[1]):int(start_position[1])+temp_img_shape[1],
                int(start_position[2]):int(start_position[2])+temp_img_shape[2]] = temp_img

        new_mix_label[int(start_position[0]):int(start_position[0])+temp_img_shape[0],
                int(start_position[1]):int(start_position[1])+temp_img_shape[1],
                int(start_position[2]):int(start_position[2])+temp_img_shape[2]] = temp_img_label

    
    tifffile.imwrite(_PATH_DATA+f"/artificial_mixes_256/mix_{str(i).zfill(5)}.tif", new_mix)
    tifffile.imwrite(_PATH_DATA+f"/artificial_mixes_256/label_{str(i).zfill(5)}.tif", new_mix_label)

def generate_mix(df, i):
    bin = get_bin(df)
    create_mix(bin, i)


if __name__=="__main__":

    df = pd.read_csv("/".join([_PATH_DATA,"bugnist_256_cut/sizes.csv"]))
    
    Parallel(n_jobs=-1)(delayed(generate_mix)(df, i) for i in tqdm(range(10000), unit="image", desc="creating mixed images"))
                               

    


    