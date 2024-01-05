# %%
import os
import subprocess
from pathlib import Path
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
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from classes import cls_attr_dist, class_names

# %%
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

# %%
# Modified PCDet functions:
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
        try:
            box_list = boxes_lidar_to_nusenes(det)
            box_list = lidar_nusc_box_to_global(
                nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
            )
        except:
            print("Typeerror: string indices must be integers")
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


# %%
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

# %%
def get_sample_token(fn):
    with open("token_dict.json", 'r') as f:
        token_dict = json.load(f)
    sample_token = token_dict[fn]['sample_token']
    f.close()
    return sample_token

# Read json file:
def read_preds_file(fn):
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

def read_results():
    """
    Reads the results from prediction files.

    Returns:
        list: A list of dictionaries containing the processed prediction data.
    """
    preds_fn = os.listdir(preds_dir)
    results = []
    count = 1
    for fn in preds_fn[:2]:
        if count%1000 == 0:
            print("Read results count: ", str(count))
        results.append(read_preds_file(fn))
        count += 1
    return results

def custom_result(fn):
    results = [read_preds_file(fn)]
    return results


# %%
def save_nusc_results(det_annos, **kwargs):
    nusc_annos = transform_det_annos_to_nusc_annos(det_annos, nusc)
    nusc_annos['meta'] = {
        'use_camera': False,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False,
    }

    output_path = Path(kwargs['output_path'])
    output_path.mkdir(exist_ok=True, parents=True)
    res_path = str(output_path / 'results_nusc.json')
    with open(res_path, 'w') as f:
        json.dump(nusc_annos, f)
    
    print('The predictions of NuScenes have been saved to {res_path}')
    return output_path, res_path

def get_metrics(output_path, res_path):
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
        'v1.0-test': 'test'
    }
    try:
        eval_version = 'detection_cvpr_2019'
        eval_config = config_factory(eval_version)
    except:
        eval_version = 'cvpr_2019'
        eval_config = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=eval_config,
        result_path=res_path,
        eval_set=eval_set_map['v1.0-test'],
        output_dir=str(output_path),
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

    with open(output_path / 'metrics_summary.json', 'r') as f:
        metrics = json.load(f)
    return metrics, metrics_summary, nusc_eval

# %%
# construct_token_dict()
results = read_results()
nusc_results = transform_det_annos_to_nusc_annos(results, nusc)
nusc_results

# %%
print(len(nusc_results))
output_path, res_path = save_nusc_results(results, output_path="/home/apurvabadithela/software/run_nuscenes_evaluations")
metrics, metrics_summary = get_metrics(output_path, res_path)

# %%



