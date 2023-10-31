import torch
from src.data.dataloaders import  CycleGANDataModule
from src.models.CycleGan import CycleGan
import tifffile
from glob import glob
from src import _PATH_DATA, _PATH_MODELS
import os
from tqdm import tqdm

def toImage(x):
    x = x.squeeze()
    x = x.numpy()
    x = x.astype('uint8')
    return x
    
def run_model(model,file):
    os.makedirs("/".join([_PATH_DATA,"synthetic_gan","/".join(file.split("/")[1:-1])]), exist_ok=True)
    img = tifffile.imread("/".join([_PATH_DATA,file]))
    img = torch.Tensor(img)
    img = img.unsqueeze(dim=0)
    img = img.unsqueeze(dim=0)
    img = img.to(torch.float)
    img = img.to("cuda")
    
    with torch.no_grad():
        fake = torch.nn.functional.sigmoid(model.genX(img))*255

    tifffile.imwrite("/".join([_PATH_DATA,"synthetic_gan","/".join(file.split("/")[1:])]),toImage(fake.cpu()))

if __name__ == "__main__":
    model = CycleGan()
    model_path = "/".join([_PATH_MODELS,"CycleGAN_UNet-2023-10-19-0921/CycleGAN-epoch=29.ckpt"])
    model.load_state_dict(torch.load(model_path, map_location=None)['state_dict'], strict=True)
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)
    model = model.cuda()
    model.eval();

    train_files = glob("synthetic_mixed_256/train/**/mix*.tif",root_dir=_PATH_DATA)
    test_files = glob("synthetic_mixed_256/test/**/mix*.tif",root_dir=_PATH_DATA)
    val_files = glob("synthetic_mixed_256/validation/**/mix*.tif",root_dir=_PATH_DATA)
    
    for file in tqdm(train_files, unit="image", desc="generating images"):
        run_model(model, file)

    for file in tqdm(test_files, unit="image", desc="generating images"):
        run_model(model, file)

    for file in tqdm(val_files, unit="image", desc="generating images"):
        run_model(model, file)
        
