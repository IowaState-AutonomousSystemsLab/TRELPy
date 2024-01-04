import os
import subprocess
from pathlib import Path
import pdb
import torch
from mmdet3d.evaluation.metrics import nuscenes_metric as nus_metric
from mmdet3d.evaluation.metrics.nuscenes_metric import output_to_nusc_box
import json 

backend_args = None
home_dir = str(Path.home())
dataroot = f"{home_dir}/software/mmdetection3d/data/nuscenes/"
out_dir = f"{home_dir}/nuscenes_dataset/inference_results"
ann_file=dataroot + 'nuscenes_infos_val.pkl'
metric='bbox'

pcd_path = f"{home_dir}/software/mmdetection3d/data/nuscenes/samples/LIDAR_TOP/"
mmdet_path = f"{home_dir}/software/mmdetection3d"
pcd_list = os.listdir(pcd_path)

# Config and model:
configs_path = "configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
checkpoint_path = "checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"

# Instantiate evaluator:
evaluator = nus_metric.NuScenesMetric(dataroot, ann_file)

# Read json file:
preds_dir = os.path.join(out_dir, "preds")
preds_fn = os.listdir(preds_dir)
for fn in preds_fn[:2]:
    fn = os.path.join(preds_dir, fn)
    print(fn)
    with open(fn, 'r') as f:
        result = json.load(f)
        result['scores_3d'] = torch.Tensor(result['scores_3d'])
        result['labels_3d'] = torch.Tensor(result['labels_3d'])
        
# Convert the results:
