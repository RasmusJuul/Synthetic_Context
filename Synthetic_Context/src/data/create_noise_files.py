import tifffile
import numpy as np
from src import _PROJECT_ROOT, _PATH_DATA
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob

def random_crop(i, file):
    img = tifffile.imread(file)
    h1 = np.random.randint(50,850)
    w1 = np.random.randint(50,600)
    d1 = np.random.randint(50,600)
    h2 = h1 + np.random.randint(10,60)
    w2 = w1 + np.random.randint(10,60)
    d2 = d1 + np.random.randint(10,60)

    new_img = img[h1:h2,w1:w2,d1:d2]
    
    n = len(glob(_PATH_DATA+f"/noise/{i}_*.tif"))
    
    tifffile.imwrite(_PATH_DATA+f"/noise/{i}_{str(n).zfill(3)}.tif",new_img)


if __name__=="__main__":
    blad = glob(_PATH_DATA+"/mixed_crop/Blad/*.tif")
    papir = glob(_PATH_DATA+"/mixed_crop/papir_fluf/*.tif")
    savs = glob(_PATH_DATA+"/mixed_crop/savs_fluf/*.tif")
    noise = blad + papir + savs

    np.random.seed(42)

    for j in range(50):
        Parallel(n_jobs=-1)(delayed(random_crop)(i,file) for i,file in tqdm(enumerate(noise), unit="image", desc="cropping images"))


