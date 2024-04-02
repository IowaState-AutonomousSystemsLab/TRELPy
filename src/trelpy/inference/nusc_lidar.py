#! /usr/bin/python3

"""This file is used to run inference on the lidar point cloud data using the trained model on the nuScenes dataset

This file requires the following environment:
    * custom_env.py be setup correctly
    * mmdetection3d be installed correctly
    * nuScenes dataset be downloaded and the path be set in accordance to mmdetection3d requirements (https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html)
"""

import os
import sys
import subprocess
from pathlib import Path
from common_imports import time_at_run, nusc
from config import config_path, checkpoint_path, dataset_root, res_dir, DET_THRESH
from trelpy.utils.mmdet_to_nusc_converter import construct_token_dict, read_results, filter_results, transform_det_annos_to_nusc_annos, save_nusc_results, get_metrics


home_dir = str(Path.home())
folder_name = f"model_{time_at_run}"
inference_res_dir = f"{res_dir}/model_{time_at_run}"

if not os.path.exists(inference_res_dir):
    os.mkdir(inference_res_dir)

info_file = os.path.join(inference_res_dir, "model_info.txt")
with open(info_file, 'w') as f:
    f.write(f"configs_path = {config_path} \n checkpoint_path = {checkpoint_path} \n")
f.close()
    
pcd_path = f"{dataset_root}/samples/LIDAR_TOP/"

pcd_list = os.listdir(pcd_path)
print(len(pcd_list))

for i, pcd in enumerate(pcd_list):
    pcd_file_path = str(Path(f"{pcd_path}/{pcd}").absolute())
    
    if pcd_file_path.exists():
        cmd = f'python3 demo/pcd_demo.py {pcd_file_path} {config_path} {checkpoint_path} --device cuda --out-dir {inference_res_dir} --pred-score-thr {DET_THRESH}'
    else:
        print(f"File {pcd_file_path} does not exist. Exiting...")
        sys.exit(1)
        
    subprocess.run(cmd, cwd=f"{home_dir}/software/mmdetection3d/", shell=True)
    
    if i%10000 == 0:
        print(f"---- ---- !-!-!-!- run_inference.py: Done with {i} files")

with open(info_file, 'a') as f:
    f.write(f"Inference for {i} files complete.")
f.close()

print(" !-!-!-!- nusc_lidar.py COMPLETE")



construct_token_dict(inference_res_dir)
results = read_results(inference_res_dir)
results = filter_results(results)
nusc_results = transform_det_annos_to_nusc_annos(results, nusc)

output_path, res_path = save_nusc_results(results, output_path=inference_res_dir)
metrics, metrics_summary, obj = get_metrics(output_path, res_path)

#TODO confirm this works