import os
import pdb
import subprocess

# Choosing a file
from pathlib import Path
from nuscenes.nuscenes import NuScenes

home_dir = str(Path.home())
out_dir = f"{home_dir}/nuscenes_dataset/inference_results"
pcd_path = f"{home_dir}/nuscenes_dataset/nuscenes_full/samples/LIDAR_TOP/"
print

pcd_list = os.listdir(pcd_path)

nusc = NuScenes(version='v1.0-trainval', dataroot = f"{home_dir}/nuscenes_dataset/nuscenes_full")

for scene in nusc.scene:
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        lidar_data = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        filename = lidar_data['filename']
        print(filename)
        sample_toke = nusc.get('sample_data', sample_token)
