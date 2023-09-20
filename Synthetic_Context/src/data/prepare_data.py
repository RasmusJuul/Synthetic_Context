import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
import scipy.ndimage as ndi
from tifffile import imwrite
from joblib import Parallel, delayed

def create_mask(im,intensity_threshold,iteration):
    h,_,_ = ndi.center_of_mass(im)
    h = int(h)
    
    bottom = 0
    for i in range(max(h-(20+iteration),0),0,-1):
        if im[i,:,:].max() < intensity_threshold:
            bottom = i
            break
    
    top = im.shape[0]
    for i in range(min(h+(20+iteration),im.shape[0]),im.shape[0]):
        if im[i,:,:].max() < intensity_threshold:
            top = i
            break

    mask = np.zeros(im.shape)
    mask[bottom:top,:,:] = 1
    
    im_avg = ndi.convolve(im, np.ones((3,3,3))/(3**3))
    
    if im[mask == 1].max() < im_avg.max():
        im[mask == 1] = 0
        return create_mask(im,intensity_threshold,iteration+1)
    return mask

def cut(file):
    im = io.imread(file)
    mask = create_mask(im, 80, 0)
    im[mask == 0] = 0
    im[im <= 40] = 0
    
    angle1 = np.random.randint(0,360)
    angle2 = np.random.randint(0,360)
    angle3 = np.random.randint(0,360)
    
    out = ndi.rotate(im, angle1, axes = (1,0))
    out = ndi.rotate(out, angle2, axes = (1,2))
    out = ndi.rotate(out, angle3, axes = (0,2))

    ax2_top = out.shape[2]
    ax2_bottom = 0
    ax1_top = out.shape[1]
    ax1_bottom = 0
    ax0_top = out.shape[0]
    ax0_bottom = 0
    
    for i in range(out.shape[2]):
        if np.any(out[:,:,i] >= 75):
            ax2_bottom = i
            break
    for i in range(out.shape[2]-1,-1,-1):
        if np.any(out[:,:,i] >= 75):
            ax2_top = i
            break
    for i in range(out.shape[1]):
        if np.any(out[:,i,:] >= 75):
            ax1_bottom = i
            break
    for i in range(out.shape[1]-1,-1,-1):
        if np.any(out[:,i,:] >= 75):
            ax1_top = i
            break
    for i in range(out.shape[0]):
        if np.any(out[i,:,:] >= 75):
            ax0_bottom = i
            break
    for i in range(out.shape[0]-1,-1,-1):
        if np.any(out[i,:,:] >= 75):
            ax0_top = i
            break
            
    cut = out[ax0_bottom:ax0_top,ax1_bottom:ax1_top,ax2_bottom:ax2_top]

    temp = file.split("/")
    temp[-3] = temp[-3] + "_cut"
    new_filename = "/".join(temp)
    imwrite(new_filename, cut)



if __name__=="__main__":
    class_folders = os.listdir("../../data/bugnist_256")
    for folder in class_folders:
        os.makedirs(f"../../data/bugnist_256_cut/{folder}",exist_ok=True)
        os.makedirs(f"../../data/bugnist_128_cut/{folder}",exist_ok=True)
        
    np.random.seed(42)
    
    for size in ["256","128"]:
        image_paths = glob.glob(f"../../data/bugnist_{size}/**/*.tif")
        Parallel(n_jobs=-1)(delayed(cut)(image_path) for image_path in tqdm(image_paths, unit="image", desc="cutting images"))
    


