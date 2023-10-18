from glob import glob
import numpy as np
import scipy.ndimage as ndi
import tifffile
from src import _PATH_DATA
from joblib import Parallel, delayed
import os
from tqdm import tqdm

def calculate_distances(file):
    img = tifffile.imread("/".join([_PATH_DATA,file]))
    img = img.astype("int16")
    dists = np.zeros((13,256,128,128))
    img[img == 0] = -1
    for i in range(1,13):
        img[img == i] = 0
        dists[i,:,:,:] = ndi.distance_transform_cdt(img)
        img[img == 0] = i

    dists[0,:,:,:] = np.inf
    dists[dists == -1] = np.inf
    dists = np.argmin(dists,axis=0)
    os.makedirs(f"{_PATH_DATA}/metric_distances/{file.split('/')[-2]}", exist_ok=True)
    
    tifffile.imwrite(f"{_PATH_DATA}/metric_distances/{file.split('/')[-2]}/{file.split('/')[-1][:-10]}_distances.tif",dists.astype("uint8"))

if __name__ == "__main__":
    files = glob("metric_labels/**/*.tif",root_dir=_PATH_DATA)

    Parallel(n_jobs=-1)(delayed(calculate_distances)(file) for file in tqdm(files, unit="image", desc="calculating distances"))