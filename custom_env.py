# #! /usr/bin/python3
raise NameError("CUSTOM ENV.PY SHOULD NOT BE CALLED. TRANSITION TO CONFIG.PY")

# import os
# import sys
# import subprocess
# from pathlib import Path
# import subprocess
# from nuscenes.eval.common.config import config_factory

# ######## PARMS #########
# ## ONLY Change model_name here to make it work with your version.
# # DET_THRESH = 0.1 # Default is 0.35 if not indicated in model name
# # model_name = f"model_good_dtr_{DET_THRESH}" 
# model_name = "model_good"
# modality = "lidar"
# is_mini = True
# ########################
# #### Get Repo Root #####
# def getGitRoot():
#     return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

# def create_dir_if_not_exist(dir_path):
#     if not os.path.exists(dir_path):
#         print(f"Directory {dir_path} not found. Creating...")
#         os.makedirs(dir_path)
#     else:
#         print(f"Not creating {dir_path} because it already exists")

# def is_set_to_mini():
#     return is_mini

# if is_mini:
#     dataset = "nuscenes" # When running run_inference.py, set to nuscenes
#     size = "mini"
#     inf_res = "inference_results_mini"
# else:
#     dataset = "nuscenes"
#     size= "full"
#     inf_res = "inference_results"

# home_dir = str(Path.home())
# dataset_root = f"{home_dir}/software/mmdetection3d/data/{dataset}/"
# output_dir = f"{home_dir}/nuscenes_dataset/{inf_res}"
# model_dir = str(Path(f"{output_dir}/{model_name}").absolute())
# preds_dir = str(Path(f"{model_dir}/preds").absolute())
# repo_dir = getGitRoot()
# cm_dir = str(Path(f"{repo_dir}/saved_cms/{modality}/{size}/{model_name}").absolute())
# create_dir_if_not_exist(cm_dir)

# ###########################
# ### Standard Parameters ###
# eval_set_map = {
#         'v1.0-mini': 'mini_val',
#         'v1.0-trainval': 'val',
#         'v1.0-test': 'test'
#     }

# dataset_version = 'v1.0-mini' if is_set_to_mini() else 'v1.0-trainval'

# try:
#     eval_version = 'detection_cvpr_2019'
#     eval_config = config_factory(eval_version)
# except:
#     eval_version = 'cvpr_2019'
#     eval_config = config_factory(eval_version)