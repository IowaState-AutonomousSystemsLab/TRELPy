# Look for this filename n015-2018-10-08-16-03-24+0800__LIDAR_TOP__1538986186697691.pcd

FILENAME = "n015-2018-10-08-16-03-24+0800__LIDAR_TOP__1538986186697691.pcd"

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
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
    while sample['next'] != "":
        filename = lidar_data['filename']
        
        if FILENAME in filename:
            nusc.render_sample_data(lidar_data['token'])
            break
        sample = nusc.get("sample", sample['next'])
        lidar_data = nusc.get('sample_data', sample['data']["LIDAR_TOP"])