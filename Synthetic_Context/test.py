from src.models.unet import UNet_pl
import torch
import tifffile
import torch.nn.functional as F
import numpy as np

model=UNet_pl(
    spatial_dims=3,
    in_channels=1,
    out_channels=13,
    channels=(4, 8, 16, 32, 64),
    strides=(2, 2, 2, 2),
    lr=1e-4
)

model_path = "models/2023-09-19-1659/UNet-epoch=221.ckpt"
model.load_state_dict(torch.load(model_path, map_location=None)['state_dict'], strict=True)
model = model.to("cuda")
model.eval()

i=9
img = tifffile.imread(f"data/mixed_crop/Mix 10/Mix 10_00{i}.tif")
img = img[18:-18,21:-21,21:-21]
tifffile.imwrite(f"mix_test/Mix_10_00{i}_cut.tif",img)
img = torch.Tensor(img)
img = img.unsqueeze(0)
img = img.unsqueeze(0)
img = img.to("cuda")

with torch.no_grad():
    out = model(img)

logits = F.log_softmax(input=out, dim=1)
logits = logits.argmax(dim=1)

tifffile.imwrite(f"mix_test/Mix_10_00{i}_labelled.tif",logits.to(torch.uint8).cpu().detach().numpy())