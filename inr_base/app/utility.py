import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
from torch import nn
import math
from torchvision import transforms
import os
import glob
from PIL import Image
from torchvision.datasets import ImageFolder
import torch.nn.functional as F




class preimg(Dataset):
    def __init__(self, root_dir, resize=None):
        self.paths = glob.glob(os.path.join(root_dir, '*.png'))
        self.resize = resize
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.resize:
            img = img.resize(self.resize, Image.BILINEAR)

        W, H = img.size
        #grid
        xs = torch.linspace(0, 1, W)
        ys = torch.linspace(0, 1, H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

        pix = self.to_tensor(img)            
        pix = pix.permute(1,2,0).view(-1,3) * 2 - 1 

        # return coords, pix, (W, H)
        return {"query": coords, "gt": pix, "size":(W,H)}
    

def recons(preds, W, H):
    preds = preds.permute(0, 2, 1)
    B, N, C = preds.shape

    img = preds.view(B, W, H, C)

    img = img.permute(0, 3, 1, 2)
    img = (img + 1.0) / 2.0
    return img.clamp(0.0, 1.0)


def save_images(img_tensor, output_dir, prefix = "pred"):

    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    B = img_tensor.shape[0]
    for i in range(B):
        img = img_tensor[i].cpu()
        pil = to_pil(img)
        filename = f"{prefix}_{i:03d}.png"
        pil.save(os.path.join(output_dir, filename))