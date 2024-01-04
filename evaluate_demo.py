import os
import pdb
import json 
import torch
import operator
import subprocess

import numpy as np
from pathlib import Path
from functools import reduce
from pyquaternion import Quaternion
from classes import cls_attr_dist, class_names

from typing import Dict, List

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from mmdet3d.evaluation.metrics import nuscenes_metric as nus_metric
from mmdet3d.evaluation.metrics.nuscenes_metric import output_to_nusc_box

backend_args = None
home_dir = str(Path.home())
nusc = NuScenes(version='v1.0-trainval', dataroot = f"{home_dir}/software/mmdetection3d/data/nuscenes")
dataroot = f"{home_dir}/software/mmdetection3d/data/nuscenes/"
out_dir = f"{home_dir}/nuscenes_dataset/inference_results"
preds_dir = os.path.join(out_dir, "preds")
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
def boxes_lidar_to_nusenes(det_info: Dict) -> List:
    """
    Converts lidar boxes to NuScenes format.

    Args:
        det_info (dict): A dictionary containing lidar detection information with keys 'bboxes_3d', 'scores_3d', and 'labels_3d'.

    Returns:
        list: A list of Box objects representing the lidar boxes in NuScenes format.
    """
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

def lidar_nusc_box_to_global(nusc: NuScenes, boxes, sample_token) -> List:
    """
    Converts lidar boxes from the NuScenes coordinate system to the global coordinate system.

    Args:
        nusc (NuScenes): An instance of the NuScenes class.
        boxes (list): A list representing lidar boxes. Boxes are in <UNKNOWN> coordinate system
        sample_token (str): The token of the sample containing the lidar data.

    Returns:
        list: A list representing the lidar boxes in the global coordinate system.
    """
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

def transform_det_annos_to_nusc_annos(det_annos: List, nusc: NuScenes) -> Dict:
    """
    Transforms detection annotations to NuScenes annotations.

    Args:
        det_annos (list): A list of detection annotations.
        nusc (NuScenes): An instance of the NuScenes class.

    Returns:
        dict: A dictionary containing the transformed NuScenes annotations with keys 'results' and 'meta'. 
        The resulting annotations have a confidence level greater than 0.6

    Raises:
        KeyError: If a required key is missing in the detection annotations.

    """
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
            if det['scores_3d'][k] >= 0.6:
                annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos

def get_sample_token(fn: str) -> Dict:
    """
    Gets the sample token associated with a given filename.

    Args:
        fn (str): The filename for which to retrieve the sample token.

    Returns:
        Dict: The sample token associated with the given filename.

    """
    with open("token_dict.json", 'r') as f:
        token_dict = json.load(f)
    sample_token = token_dict[fn]['sample_token']
    f.close()
    return sample_token

# Read json file:
def read_preds_file(fn: str) -> Dict:
    """
    Reads the predictions file and processes the data.

    Args:
        fn (str): The filename of the predictions file.

    Returns:
        dict: A dictionary containing the processed prediction data with keys 'bboxes_3d', 'scores_3d', 'labels_3d', 'metadata', and 'name'.

    """
    full_fn = os.path.join(preds_dir, fn)
    with open(full_fn, 'r') as f:
        result = json.load(f)
        result['bboxes_3d'] = torch.Tensor(result['bboxes_3d']).numpy()
        result['scores_3d'] = torch.Tensor(result['scores_3d']).numpy()
        class_labels = [class_names[k] for k in result['labels_3d']]
        result['labels_3d'] = torch.Tensor(result['labels_3d']).numpy()
        sample_token = dict()
        sample_token['token'] = get_sample_token(fn)
        result.update({'metadata':sample_token})
        result.update({'name':class_labels})
        
    f.close()
    return result
        # nusc_box = output_to_nusc_box(result)

def read_results():
    """
    Reads the results from prediction files.

    Returns:
        list: A list of dictionaries containing the processed prediction data.
    """
    preds_fn = os.listdir(preds_dir)
    results = []
    for fn in preds_fn:
        results.append(read_preds_file(fn))
    return results

def construct_token_dict():
    """
    Constructs a dictionary mapping filenames to lidar and sample tokens
    """
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
        
        filename = lidar_data['filename']
        file_str = filename[filename.rfind("/")+1:].replace("bin", "json")
        token_dict[file_str] = {"lidar_token": lidar_data['token'], "sample_token": lidar_data['sample_token']}

    with open("token_dict.json", 'w') as f:
        json.dump(token_dict, f)
    f.close()

construct_token_dict()
pdb.set_trace()
results = read_results()
nusc_results = transform_det_annos_to_nusc_annos(results, nusc)
pdb.set_trace()