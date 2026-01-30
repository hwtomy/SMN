import kagglehub
import shutil
import os


dataset_name = "nguyenhung1903/nerf-synthetic-dataset"
save_dir = "nerf_blender"  


print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download(dataset_name)
print("Downloaded to:", path)


if path != save_dir:
    if os.path.exists(save_dir):
        print(f"Removing old directory at {save_dir}")
        shutil.rmtree(save_dir)
    print(f"Copying to {save_dir} ...")
    shutil.copytree(path, save_dir)

print(" Dataset is ready at:", save_dir)
