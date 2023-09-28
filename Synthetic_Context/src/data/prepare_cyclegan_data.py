from glob import glob
from src import _PATH_DATA
import numpy as np
import tifffile
from skimage.transform import resize
from joblib import Parallel, delayed
from tqdm import tqdm


def scale_mixes(file, mode):
    temp_img = tifffile.imread("/".join([_PATH_DATA, file]))
    temp_img = resize(temp_img, (256, 128, 128))
    temp_img = np.uint8(np.clip(temp_img, 0, 1) * 255)
    tifffile.imwrite(
        "/".join([_PATH_DATA, f"cyclegan_256/{mode}B", file.split("/")[-1]]), temp_img
    )


if __name__ == "__main__":
    files = glob("mixed_crop/Mix*/Mix*.tif", root_dir=_PATH_DATA)
    files.sort()
    files = np.array(files)

    np.random.seed(42)
    idxs = np.random.choice(len(files), 100, replace=False)

    test_files = files[idxs[:50]]
    validation_files = files[idxs[50:]]
    train_files = np.delete(files, idxs)

    Parallel(n_jobs=-1)(
        delayed(scale_mixes)(file, "train")
        for file in tqdm(train_files, unit="image", desc="resizing images")
    )
    Parallel(n_jobs=-1)(
        delayed(scale_mixes)(file, "test")
        for file in tqdm(test_files, unit="image", desc="resizing images")
    )
    Parallel(n_jobs=-1)(
        delayed(scale_mixes)(file, "validation")
        for file in tqdm(validation_files, unit="image", desc="resizing images")
    )
