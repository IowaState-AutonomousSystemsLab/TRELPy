#! /usr/bin/python3

import os
import sys
from pathlib import Path
import subprocess

######## PARMS #########
## ONLY Change model_name here to make it work with your version.
model_name = "model_01-05-2024_20_22"
modality = "lidar"
is_mini = True
########################
#### Get Repo Root #####
def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} not found. Creating...")
        os.makedirs(dir_path)
    else:
        print(f"Not creating {dir_path} because it already exists")

def is_set_to_mini():
    return is_mini

if is_mini:
    dataset = "nuscenes"
    size = "mini"
    inf_res = "inference_results_mini"
else:
    dataset = "nuscenes-full"
    size= "full"
    inf_res = "inference_results"

home_dir = str(Path.home())
dataset_root = f"{home_dir}/software/mmdetection3d/data/{dataset}/"
output_dir = f"{home_dir}/nuscenes_dataset/{inf_res}"
model_dir = str(Path(f"{output_dir}/{model_name}").absolute())
preds_dir = str(Path(f"{model_dir}/preds").absolute())
repo_dir = getGitRoot()
cm_dir = str(Path(f"{repo_dir}/saved_cms/{modality}/{size}/{model_name}").absolute())
create_dir_if_not_exist(cm_dir)