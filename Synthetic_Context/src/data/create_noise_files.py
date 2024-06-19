import tifffile
import numpy as np
from src import _PROJECT_ROOT, _PATH_DATA
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob


def create_noise_file(i, files):
    width = 92
    noise = np.zeros((128,92,92))
    for j in range(width):
        img = tifffile.imread(np.random.choice(files))
        if np.random.randint(2):
            slice = img[:,:,np.random.randint(10,50)]*0.5
        else:
            slice = img[:,np.random.randint(10,50),:]*0.5
        offset = np.random.randint(92-64)
        noise[:,offset:92-(92-64-offset),j] = slice
    
    for j in range(width):
        img = tifffile.imread(np.random.choice(files))
        if np.random.randint(2):
            slice = img[:,:,np.random.randint(10,50)]*0.5
        else:
            slice = img[:,np.random.randint(10,50),:]*0.5
        offset = np.random.randint(92-64)
        noise[:,j,offset:92-(92-64-offset)] = slice
    
    noise += np.random.normal(loc=25,scale=10,size=(128,92,92))
    noise[noise < 0] = 0

    tifffile.imwrite(f"{_PATH_DATA}/noise/noise_{str(i).zfill(3)}.tif", noise.astype('uint8'))

def update_noise(files):
    for file in files:
        img1 = tifffile.imread(file)
        img2 = tifffile.imread(np.random.choice(files))
        noise = (img1 + img2)
        noise[noise > 100] = noise[noise > 100]*0.75
        tifffile.imwrite(f"{_PATH_DATA}/noise2/{file.split('/')[-1]}",noise.astype('uint8'))

if __name__ == "__main__":
    files = glob(f"{_PATH_DATA}/BugNIST_DATA/train/**/*.tif")
    np.random.seed(42)

    Parallel(n_jobs=-1)(delayed(create_noise_file)(i, files) for i in tqdm(range(500), unit="image", desc="creating mixed images"))

    files = glob(f"{_PATH_DATA}/noise/*.tif")
    update_noise(files)
