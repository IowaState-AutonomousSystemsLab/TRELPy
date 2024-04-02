#! /usr/bin/python3

from src.trelpy.common_imports import create_dir_if_not_exist
import subprocess
from pathlib import Path
from nuscenes.eval.common.config import config_factory


######## PARMS #########
## Inference model params ##
_config_file     = "pointpillars_hv_fpn_sbn-all_8xb2-amp-2x_nus-3d.py"  #........................# The config file for the model
_checkpoint_file = "hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth" # The checkpoint file for the model

DET_THRESH = 0.1                      # The confidence threshold to reject predictions
model_name = "pointpillars_mmdet3d"   # The name of the directory where the ML model for inference is stored
modality = "lidar"                    # The modality of the data
is_mini = True                        # Are you using this on NuScenes Mini?

## Confusion Matrix Generation Params ##
verbose = True
###### PARAMS END ######


####### Configuring the right dataset ########
# The code looks in mmdetection3d/data/ for a dataset folder or symlink called `dataset` to find a dataset with size `size`.
# The results will be stored in inside a folder titled `inference_results_path`
if is_mini:
    dataset = "nuscenes-mini"   
    size = "mini"
else:
    dataset = "nuscenes-full"
    size= "full"
    
########### METHODS ############
def is_set_to_mini():
    return is_mini
###### METHODS END #########


home_dir = str(Path.home())
repo_dir = f"{home_dir}/nuscenes_dataset/3D_Detection"  #................# The directory where the repo is stored
dataset_root = f"{repo_dir}/software/mmdetection3d/data/{dataset}"  #...# The directory where the dataset is stored
res_dir = f"{repo_dir}/results/{dataset}/{model_name}" #.................# The directory where the output of inference will be stored
mmdet3d_dir = f"{home_dir}/software/mmdetection3d" #.....................# The directory where the mmdetection3d repo is stored
config_path = f"{mmdet3d_dir}/configs/{_config_file}" #...................# The path to the config file
checkpoint_path = f"{repo_dir}/models/{_checkpoint_file}" #...............# The path to the checkpoint file

create_dir_if_not_exist(res_dir)

###########################
### Standard Parameters ###
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