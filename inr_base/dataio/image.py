import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm
import imageio
import cv2


class ImageImplicit(Dataset):
    def __init__(self, paths, image_size, n_samples_per_image, device="cpu"):
        super().__init__()

        self.paths = paths
        self.image_size = image_size
        self.n_samples_per_image = n_samples_per_image
        self.device = device

        self._load_images()


    @torch.no_grad()
    def _load_images(self):
        # print(paths)
        images = [imageio.imread(path) for path in self.paths]
        print(self.paths)
        self.original_sizes = [img.shape[:2] for img in images]

        images = [cv2.resize(image, self.image_size) for image in images]
        print(len(images))
        images = np.stack(images, axis=0)
        
        # images = np.array(images, dtype=np.float32)
        # images = np.expand_dims(images, axis=0)
        images = torch.from_numpy(images).to(self.device)
        images = images.permute(0, 3, 1, 2)
        images = images / 255.
        images = images * 2. - 1.
        self.images = images
        print(images.shape)

    def __len__(self):
        return 1000000000

    @torch.no_grad()
    def __getitem__(self, index):

        N = len(self.images)
        np.random.seed(index)

        coordinates = np.random.uniform(size=[2, self.n_samples_per_image]).astype(np.float32)
        coordinates = torch.from_numpy(coordinates).to(self.device) * 2. - 1.
        coordinates = coordinates.unsqueeze(0).repeat(N, 1, 1)

        coordinates_bi = coordinates.unsqueeze(2).permute(0, 2, 3, 1)
        rgb = F.grid_sample(self.images, coordinates_bi, mode='bilinear', align_corners=True)

        coordinates = coordinates.squeeze(2)
        rgb = rgb.squeeze(2)
        return {"query": coordinates, "gt": rgb, "size":self.original_sizes}