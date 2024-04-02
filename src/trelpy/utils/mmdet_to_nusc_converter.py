#! /usr/bin/python

"""This is a utility file meant to convert the predictions of a model trained on the mmdet3d framework to the NuScenes format.

"""

import os
import sys
import json 
import torch
import operator
import numpy as np

from pathlib import Path
from typing import Dict, List
from datetime import datetime
from pyquaternion import Quaternion

from mmdet3d.evaluation.metrics import nuscenes_metric as nus_metric

from nuscenes.utils.data_classes import Box
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval, load_gt
from nuscenes.eval.detection.data_classes import DetectionBox

from common_imports import nusc, time_at_run, create_dir_if_not_exist
from classes import cls_attr_dist, class_names, mini_val_tokens
from config import dataset_root
from config import home_dir, res_dir, is_set_to_mini, dataset_version, eval_config, eval_set_map, DET_THRESH

ann_file = f'{dataset_root}/nuscenes_infos_val.pkl'
metric='bbox'

pcd_path = f"{dataset_root}/samples/LIDAR_TOP/"
pcd_list = os.listdir(pcd_path)

# Instantiate evaluator:
evaluator = nus_metric.NuScenesMetric(dataset_root, ann_file)

# Modified PCDet functions:
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

def lidar_nusc_box_to_global(boxes, sample_token) -> List:
    """Converts lidar boxes from the NuScenes coordinate system to the global coordinate system.

    Args:
        boxes (list): A list representing lidar boxes. Boxes are in <UNKNOWN> coordinate system
        sample_token (str): The token of the sample containing the lidar data.

    Returns:
        list: A list representing the lidar boxes in the global coordinate system.
    """
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

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

def transform_det_annos_to_nusc_annos(det_annos: List) -> Dict:
    """
    Transforms detection annotations to NuScenes annotations.

    Args:
        det_annos (list): A list of detection annotations.
        nusc (NuScenes): An instance of the NuScenes class.

    Returns:
        dict: A dictionary containing the transformed NuScenes annotations with keys 'results' and 'meta'. 
        The resulting annotations have a confidence level greater than the DETECTION THRESHOLD (0.6)

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
        sys.stdout.flush()
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
            
            if det['scores_3d'][k] >= DET_THRESH:
                annos.append(nusc_anno)
            
        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos

def construct_token_dict(output_dir: str):
    """Constructs a dictionary mapping filenames to lidar and sample tokens
    """
    token_dict = {}
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

    with open(f"{output_dir}/token_dict.json", 'w') as f:
        json.dump(token_dict, f)
    f.close()


def get_sample_token(fn, inference_res_dir: str) -> Dict:
    """
    Gets the sample token associated with a given filename.

    Args:
        fn (str): The filename for which to retrieve the sample token.

    Returns:
        Dict: The sample token associated with the given filename.
    """
    with open(f"{inference_res_dir}/token_dict.json", 'r') as f:
        token_dict = json.load(f)
    sample_token = token_dict[fn]['sample_token']
    f.close()
    return sample_token


# Read json file:
def read_preds_file(json_file_path: str) -> Dict:
    """
    Reads the predictions file and processes the data.

    Args:
        fn (str): The filename of the predictions file.

    Returns:
        dict: A dictionary containing the processed prediction data with keys 'bboxes_3d', 'scores_3d', 'labels_3d', 'metadata', and 'name'.

    """
    with open(json_file_path, 'r') as f:
        result = json.load(f)
        result['bboxes_3d'] = torch.Tensor(result['bboxes_3d']).numpy()
        result['scores_3d'] = torch.Tensor(result['scores_3d']).numpy()
        class_labels = [class_names[k] for k in result['labels_3d']]
        result['labels_3d'] = torch.Tensor(result['labels_3d']).numpy()
        sample_token = {}
        sample_token['token'] = get_sample_token(fn)
        result.update({'metadata':sample_token})
        result.update({'name':class_labels})
        
    f.close()
    return result


def read_results(inference_res_dir: str):
    """Reads the results from prediction files in the given directory.

    Returns:
        list: A list of dictionaries containing the processed prediction data.
    """
    preds_filenames = os.listdir(inference_res_dir)
    results = []
    for count, fn in enumerate(preds_filenames, start=1):
        if count%1000 == 0:
            print("Read results count: ", count)
        results.append(read_preds_file(f"{inference_res_dir}/fn"))
    return results

def custom_result(fn):
    return [read_preds_file(fn)]


def save_nusc_results(det_annos, inference_res_dir:str):
    nusc_annos = transform_det_annos_to_nusc_annos(det_annos, nusc)
    nusc_annos['meta'] = {
        'use_camera': False,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False,
    }

    output_path = Path(inference_res_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    res_path = str(output_path / 'results_nusc.json')
    with open(res_path, 'w') as f:
        json.dump(nusc_annos, f)
    
    print(f'The predictions of NuScenes have been saved to {res_path}')
    return output_path, res_path

def get_metrics(output_path, res_path):
    print(f' Ln279: get_metrics called on {res_path} with {dataset_version} and {eval_set_map[dataset_version]}')
    nusc_eval = NuScenesEval(
        nusc,
        config=eval_config,
        result_path=res_path,
        eval_set=eval_set_map[dataset_version],
        output_dir=str(output_path),
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

    with open(output_path / 'metrics_summary.json', 'r') as f:
        metrics = json.load(f)
    return metrics, metrics_summary, nusc_eval

def filter_results(results):
    """
        it only val results
    """
    toks = load_gt(nusc, eval_set_map[dataset_version], DetectionBox, verbose=False).sample_tokens
    return [
        result
        for result in results
        if result['metadata']['token'] in toks
    ]