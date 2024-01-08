#! /usr/bin/python3

import os
import sys
from pathlib import Path

######## PARMS #########
model_name = "model2_good"
is_mini = True
########################

if is_mini:
    dataset = "nuscenes-mini"
    inf_res = "inference_results_mini"
else:
    dataset = "nuscenes"
    inf_res = "inference_results"


home_dir = str(Path.home())
dataset_root = f"{home_dir}/software/mmdetection3d/data/{dataset}/"
output_dir = f"{home_dir}/nuscenes_dataset/{inf_res}"
model_dir = str(Path(f"{output_dir}/{model_name}").absolute())
preds_dir = str(Path(f"{model_dir}/preds").absolute())

def is_set_to_mini():
    return is_mini