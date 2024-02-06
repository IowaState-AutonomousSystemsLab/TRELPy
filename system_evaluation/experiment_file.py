# cm_fn = "/home/apurvabadithela/software/run_nuscenes_evaluations/saved_cms/lidar/mini/cm.pkl"
# control_dir = "/home/apurvabadithela/software/run_nuscenes_evaluations/system_evaluation/controllers/"

import os
import sys
from pathlib import Path

######## PARMS #########
model_name = "model2_good"
modality = "lidar"
is_mini = True
########################

if is_mini:
    dataset = "nuscenes-mini"
    inf_res = "inference_results_mini"
else:
    dataset = "nuscenes"
    inf_res = "inference_results"

is_mini = True
def is_set_to_mini():
    return is_mini

def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} not found. Creating...")
        os.makedirs(dir_path)
    else:
        print(f"Not creating {dir_path} because it already exists")


home_dir = str(Path.home())
dataset_root = f"{home_dir}/software/mmdetection3d/data/{dataset}/"
output_dir = f"{home_dir}/nuscenes_dataset/{inf_res}"
model_dir = str(Path(f"{output_dir}/{model_name}").absolute())
preds_dir = str(Path(f"{model_dir}/preds").absolute())
repo_dir = str(Path(f"{home_dir}/nuscenes_dataset/3D_Detection").absolute())
cm_dir = str(Path(f"{repo_dir}/saved_cms/{modality}/{dataset}/{model_name}").absolute())
create_dir_if_not_exist(cm_dir)
cm_fn = f"{cm_dir}/cm.pkl"
control_dir = f"{repo_dir}/system_evaluation/controllers/"