import csv
import glob
import math
import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm


class PointCloud(Dataset):
    def __init__(
        self,
        paths,
        on_surface_points,
        keep_aspect_ratio=True,
        is_mesh=True,
        output_type="occ",
        out_act="sigmoid",
        n_points=200000,
        cfg=None,
    ):
        super().__init__()
        self.paths = paths  # List of file paths
        self.on_surface_points = on_surface_points
        self.keep_aspect_ratio = keep_aspect_ratio
        self.is_mesh = is_mesh
        self.output_type = output_type
        self.out_act = out_act
        self.n_points = n_points
        self.cfg = cfg
        self.move = cfg.mlp_config.move
        assert self.move == False
        self.total_time = 16

        self.coords_list = []
        self.occupancies_list = []

        for path in tqdm(self.paths, desc="Loading pointclouds"):

            if not cfg.in_out:
                pc_folder = os.path.dirname(path) + "_" + str(cfg.n_points) + "_pc"
            else:
                pc_folder = (
                    os.path.dirname(path)
                    + "_"
                    + str(cfg.n_points)
                    + "_pc_"
                    + f"{self.output_type}_in_out_{str(cfg.in_out)}"
                )

            if is_mesh:
                if cfg.strategy == "save_pc":
                    obj: trimesh.Trimesh = trimesh.load(path)
                    vertices = obj.vertices
                    vertices -= np.mean(vertices, axis=0, keepdims=True)
                    v_max = np.amax(vertices)
                    v_min = np.amin(vertices)
                    vertices *= 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))
                    obj.vertices = vertices
                    total_points = cfg.n_points
                    n_points_uniform = total_points
                    n_points_surface = total_points

                    points_uniform = np.random.uniform(
                        -0.5, 0.5, size=(n_points_uniform, 3)
                    )
                    points_surface = obj.sample(n_points_surface)
                    points_surface += 0.01 * np.random.randn(n_points_surface, 3)
                    points = np.concatenate([points_surface, points_uniform], axis=0)

                    inside_surface_values = igl.fast_winding_number_for_meshes(
                        obj.vertices, obj.faces, points
                    )
                    thresh = 0.5
                    occupancies_winding = np.piecewise(
                        inside_surface_values,
                        [inside_surface_values < thresh, inside_surface_values >= thresh],
                        [0, 1],
                    )
                    occupancies = occupancies_winding[..., None]
                    print(points.shape, occupancies.shape, occupancies.sum())
                    point_cloud = np.hstack((points, occupancies))
                    print(point_cloud.shape, points.shape, occupancies.shape)

                    coords = point_cloud[:, :3]
                    occupancies = point_cloud[:, 3]

                    # Save the point cloud
                    point_cloud_xyz = np.hstack((coords, occupancies[:, None]))
                    os.makedirs(pc_folder, exist_ok=True)
                    np.save(
                        os.path.join(pc_folder, os.path.basename(path)), point_cloud_xyz
                    )
                else:
                    # Load the saved point cloud
                    point_cloud = np.load(
                        os.path.join(pc_folder, os.path.basename(path) + ".npy")
                    )
                    coords = point_cloud[:, :3]
                    occupancies = point_cloud[:, 3]
            else:
                point_cloud = np.genfromtxt(path)
                coords = point_cloud[:, :3]
                occupancies = point_cloud[:, 3]

            if cfg.shape_modify == "half":
                included_points = coords[:, 0] < 0
                coords = coords[included_points]
                occupancies = occupancies[included_points]

            self.coords_list.append(coords)
            self.occupancies_list.append(occupancies)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        coords = self.coords_list[idx]
        occupancies = self.occupancies_list[idx]

        total_points = coords.shape[0]

        output_dict = {
            'x': torch.from_numpy(coords).float(),
            'y': torch.from_numpy(occupancies[:, None]).float()
        }

        return output_dict