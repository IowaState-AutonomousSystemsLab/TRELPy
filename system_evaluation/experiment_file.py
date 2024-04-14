# cm_fn = "/home/apurvabadithela/software/run_nuscenes_evaluations/saved_cms/lidar/mini/cm.pkl"
# control_dir = "/home/apurvabadithela/software/run_nuscenes_evaluations/system_evaluation/controllers/"

import os
import sys
from pathlib import Path
import subprocess
from pdb import set_trace as st

######## PARMS #########
model_name ="model_04-08-2024_09_52"
modality = "lidar"
is_mini = False
########################
def getGitRoot():
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

if is_mini:
    dataset = "nuscenes"
    size = "mini"
    inf_res = "inference_results_mini"
else:
    dataset = "nuscenes"
    size= "full"
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
repo_dir = getGitRoot()
cm_dir = str(Path(f"{repo_dir}/saved_cms/{modality}/{size}/{model_name}").absolute())
create_dir_if_not_exist(cm_dir)
cm_fn = f"{cm_dir}/low_thresh_cm.pkl"
prop_cm_fn = f"{cm_dir}/low_thresh_prop_cm.pkl"
prop_cm_seg_fn = f"{cm_dir}/low_thresh_prop_cm_cluster.pkl"
prop_dict_file = f"{cm_dir}/prop_dict.pkl"
control_dir = f"{repo_dir}/system_evaluation/controllers/"