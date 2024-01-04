import os
import subprocess
from pathlib import Path
import pdb
import torch
from mmdet3d.evaluation.metrics import nuscenes_metric as nus_metric
from mmdet3d.evaluation.metrics.nuscenes_metric import output_to_nusc_box
import json 
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
import operator
from functools import reduce
from pathlib import Path
import numpy as np
from nuscenes.nuscenes import NuScenes
from classes import cls_attr_dist

backend_args = None
home_dir = str(Path.home())
nusc = NuScenes(version='v1.0-trainval', dataroot = f"{home_dir}/software/mmdetection3d/data/nuscenes")
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

# Box conversion
def boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['bboxes_3d']
    scores = det_info['scores_3d']
    labels = det_info['labels_3d']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k], score=scores[k], velocity=velocity,
        )
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list

def transform_det_annos_to_nusc_annos(det_annos, nusc):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
        )

        for k, box in enumerate(box_list):
            name = det['name'][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos

# Read json file:
def read_preds_file(fn):
    with open(fn, 'r') as f:
        result = json.load(f)
        result['bboxes_3d'] = torch.Tensor(result['bboxes_3d']).numpy()
        result['scores_3d'] = torch.Tensor(result['scores_3d']).numpy()
        result['labels_3d'] = torch.Tensor(result['labels_3d']).numpy()
        return result
        # nusc_box = output_to_nusc_box(result)

def read_results():
    preds_dir = os.path.join(out_dir, "preds")
    preds_fn = os.listdir(preds_dir)
    results = []
    for fn in preds_fn[:2]:
        fn = os.path.join(preds_dir, fn)
        results.append(read_preds_file(fn))

# nusc_results = transform_det_annos_to_nusc_annos(results, nusc)
def construct_token_dict():
    token_dict = dict()
    for scene in nusc.scene:
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
        lidar_data = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        while sample['next'] != "":
            filename = lidar_data['filename']
            file_str = filename[filename.rfind("/")+1:].replace("bin", "json")
            token_dict[file_str] = {"lidar_token": lidar_data['token'], "sample_token": lidar_data['sample_token']}
            sample = nusc.get("sample", sample['next'])
            lidar_data = nusc.get('sample_data', sample['data']["LIDAR_TOP"])

    with open("token_dict.json", 'w') as f:
        json.dump(token_dict, f)

