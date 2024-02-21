import os
import pdb
import json
import subprocess

# Choosing a file
from pathlib import Path
from nuscenes.nuscenes import NuScenes

home_dir = str(Path.home())
nusc = NuScenes(version='v1.0-trainval', dataroot = f"{home_dir}/software/mmdetection3d/data/nuscenes")

# Get all the filenames in the nuScenes_full/samples/LIDAR_TOP folder
pcd_path = f"{home_dir}/software/mmdetection3d/data/nuscenes/samples/LIDAR_TOP/"
pcd_files = os.listdir(pcd_path)

# get all the files in the token_dict.json file
with open("token_dict.json", "r") as f:
    token_dict = json.load(f)
    json_files = list(token_dict.keys())


# get the difference between the two lists
diff = [x for x in pcd_files if x.replace("bin", "json") not in json_files]
print(f"There are {len(diff)} different1 files")

# print each element of diff on a different line
for i, elem in enumerate(diff):
    print(i, elem)

# # Get all the files in inference_results
# inference_path = f"{home_dir}/nuscenes_dataset/inference_results/"
# inference_files = os.listdir(inference_path)


