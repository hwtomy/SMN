import torch
import os
import json
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class DynamicRayDataset(Dataset):
    def __init__(self, root_dir, image_size=(200, 200), rays_per_batch=4096, device="cuda"):
        json_path = os.path.join(root_dir, "transforms_train.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)
        self.frames = meta["frames"]
        self.camera_angle_x = float(meta["camera_angle_x"])
        self.W, self.H = image_size
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)
        self.device = device
        self.rays_per_batch = rays_per_batch

        self.images = []
        self.c2ws = []
        for frame in self.frames:
            path = os.path.join(root_dir, frame["file_path"] + ".png")
            img = Image.open(path).convert("RGB").resize((self.W, self.H), Image.LANCZOS)
            img = TF.to_tensor(img).to(device) * 2. - 1.  # (3, H, W)
            self.images.append(img)
            self.c2ws.append(torch.FloatTensor(frame["transform_matrix"]).to(device))
        self.images = torch.stack(self.images)  # (N_img, 3, H, W)
        self.c2ws = torch.stack(self.c2ws)      # (N_img, 4, 4)
        self.num_images = len(self.images)

    def __len__(self):
        # Large enough for effectively "infinite" iterations
        return 1000000000

    def __getitem__(self, idx):
 
        np.random.seed(idx)


        img_idx = np.random.randint(self.num_images)
        img = self.images[img_idx]    # (3, H, W)
        c2w = self.c2ws[img_idx]      # (4, 4)
        H, W, focal = self.H, self.W, self.focal


        xs = np.random.randint(0, W, size=self.rays_per_batch)
        ys = np.random.randint(0, H, size=self.rays_per_batch)

        i = torch.from_numpy(xs).float().to(self.device)
        j = torch.from_numpy(ys).float().to(self.device)

        dirs = torch.stack([
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i)
        ], -1)  # (rays_per_batch, 3)

        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # (rays_per_batch, 3)
        rays_o = c2w[:3, 3].expand(rays_d.shape)                  # (rays_per_batch, 3)
        rgb = img[:, ys, xs].permute(1, 0)                        # (rays_per_batch, 3)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "gt": rgb
        }

class AllRaysNeRFDataset(Dataset):
    def __init__(self, root_dir, image_size=(200, 200), device="cpu"):
        json_path = os.path.join(root_dir, "transforms_train.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)
        frames = meta["frames"]
        camera_angle_x = float(meta["camera_angle_x"])
        W, H = image_size
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        all_rays_o = []
        all_rays_d = []
        all_rgbs = []

        for frame in frames:

            path = os.path.join(root_dir, frame["file_path"] + ".png")
            img = Image.open(path).convert("RGB").resize(image_size[::-1], Image.LANCZOS)
            img = TF.to_tensor(img).to(device) * 2. - 1.   # (3, H, W)
            img = img.permute(1, 2, 0).reshape(-1, 3)      # (H*W, 3)


            i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            i = torch.from_numpy(i).reshape(-1).float().to(device)
            j = torch.from_numpy(j).reshape(-1).float().to(device)

            dirs = torch.stack([
                (i - W * 0.5) / focal,
                -(j - H * 0.5) / focal,
                -torch.ones_like(i)
            ], -1)  # (H*W, 3)

            c2w = torch.FloatTensor(frame["transform_matrix"]).to(device)
            rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # (H*W, 3)
            rays_o = c2w[:3, 3].expand(rays_d.shape)                  # (H*W, 3)

            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)
            all_rgbs.append(img)

        self.rays_o = torch.cat(all_rays_o, dim=0)    # (N_total, 3)
        self.rays_d = torch.cat(all_rays_d, dim=0)    # (N_total, 3)
        self.gt = torch.cat(all_rgbs, dim=0)          # (N_total, 3)

    def __len__(self):
        return self.rays_o.shape[0]

    def __getitem__(self, idx):
        np.random.seed(idx)
        return {
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
            "gt": self.gt[idx]
        }