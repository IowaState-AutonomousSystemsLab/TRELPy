from generate_confusion_matrix import GenerateConfusionMatrix
from confusion_matrix import ConfusionMatrix
import os
import numpy as np
from typing import Tuple, Dict, Any, List
from itertools import chain, combinations
from pdb import set_trace as st

from custom_env import dataset_root as dataroot

from mmdet3d.evaluation.metrics import nuscenes_metric as nus_metric
from custom_env import home_dir, cm_dir, repo_dir, output_dir, preds_dir, model_dir, is_set_to_mini


from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes

# parameters to setup nuScenes

eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
        'v1.0-test': 'test'
    }

dataset_version = 'v1.0-mini' if is_set_to_mini() else 'v1.0-trainval'
try:
    eval_version = 'detection_cvpr_2019'
    eval_config = config_factory(eval_version)
except:
    eval_version = 'cvpr_2019'
    eval_config = config_factory(eval_version)

nusc = NuScenes(version=dataset_version, dataroot = dataroot)

list_of_classes = ["ped", "obs"]

PED = 0
OBS = 1
EMPTY = 2

labels = {0: "ped", 1: "obs", 2:"empty"}

conf_mat_mapping = {
    "pedestrian": PED,
    "bus": OBS,
    "car" : OBS,
    "truck": OBS,
    "bicycle": OBS,
    "motorcycle": OBS,
    "traffic_cone": OBS
}

generator = GenerateConfusionMatrix(nusc=nusc,
    config=eval_config,
    result_path=f'{model_dir}/results_nusc.json',
    eval_set=eval_set_map[dataset_version],
    output_dir=os.getcwd(),
    verbose=True,
    conf_mat_mapping=conf_mat_mapping,
    list_of_classes=list_of_classes,
    distance_parametrized=True,
    max_dist=100,
    distance_bin=10
)

cm_prop = generator.get_proposition_labelled_conf_mat()
cm = generator.get_distance_param_conf_mat()
x = generator.get_clustered_conf_mat()
print()


confusion_matrix = ConfusionMatrix(generator, list_of_classes, labels)
confusion_matrix.set_confusion_matrix(cm, label_type="class")
cm_file = f"{cm_dir}/cm.pkl"
confusion_matrix.save_confusion_matrix(cm_file, label_type="class")
prop_cm_file = f"{cm_dir}/prop_cm.pkl"
confusion_matrix.set_confusion_matrix(cm_prop, label_type="prop")
confusion_matrix.save_confusion_matrix(prop_cm_file, label_type="prop")
