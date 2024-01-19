# Script to generate confusion matrix for class labels
# Copy of the evaluate.py script of nuscenes
import argparse
import json
import os
import time
import random
# import pytorch3d
import numpy as np
from typing import Callable
from typing import Tuple, Dict, Any, List
from pyquaternion import Quaternion
from itertools import chain, combinations


from custom_env import dataset_root as dataroot
from classes import cls_attr_dist, class_names, mini_val_tokens
from mmdet3d.evaluation.metrics import nuscenes_metric as nus_metric
from custom_env import home_dir, output_dir, preds_dir, model_dir, is_set_to_mini


from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionMetricData
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList

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

DETECTION_THRESHOLD = 0.35

backend_args = None
nusc = NuScenes(version=dataset_version, dataroot = dataroot)
ann_file = f'{dataroot}nuscenes_infos_val.pkl'
metric='bbox'

pcd_path = f"{dataroot}/samples/LIDAR_TOP/"
mmdet_path = f"{home_dir}/software/mmdetection3d"
pcd_list = os.listdir(pcd_path)

PED = 0
OBS = 1
EMPTY = 2

distance_param_conf_mat = np.zeros((3, 3))

conf_mat_mapping = {
    "pedestrian": PED,
    "bus": OBS,
    "car" : OBS,
    "truck": OBS,
    "bicycle": OBS,
    "motorcycle": OBS,
    "traffic_cone": OBS
}

class GenConfMatrix:
    """
    This class instantiates a class-labeled confusion matrix.
    The methods in this class are used to construct a class-labeled confusion matrix for a 
    specific model on the NuScenes dataset.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 
                 distance_parametrized: bool = False,
                 lower_thresh:float = -1, 
                 upper_thresh:float = np.inf,
                 distance_bin:float = 10
                 ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        :param distance_parametrized: Whether the confusion matrix is parametrized by distance or not.
        :param lower_thresh: Lower distance threshold.
        :param upper_thresh: Upper distance threshold.
        :param distance_bin: If lower_thresh = -1 and upper_thresh = inf, there is only one confusion matrix.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        self.distance_parametrized = distance_parametrized
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        self.distance_bin = distance_bin

        self.check_distance_param_settings()

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)
        
        if verbose:
            print('Removing samples outside of distance range')
        
        for gt in self.gt_boxes.all:
            dist = np.sqrt(np.dot(gt.ego_translation, gt.ego_translation)) 
            if dist > upper_thresh or dist < lower_thresh:
                self.gt_boxes.all.remove(gt)
                self.gt_boxes.sample_tokens.remove(gt.sample_token)
        
        for pred in self.pred_boxes.all:
            dist = np.sqrt(np.dot(pred.ego_translation, pred.ego_translation)) 
            if dist > upper_thresh or dist < lower_thresh:
                self.pred_boxes.all.remove(pred)
                self.pred_boxes.sample_tokens.remove(pred.sample_token)

        self.sample_tokens = self.gt_boxes.sample_tokens


    def calculate_distance_param_conf_mat(gt_boxes:EvalBoxes, 
                                      pred_boxes: EvalBoxes, 
                                      conf_mat_mapping: Dict,
                                      dist_thresh: float = 2.0,       # in m 
                                      yaw_thresh: float = np.pi/2.0): # in radians  -> np.ndarray:
    
        
        c = 0
        # -- For each sample
        # -- -- For each ground truth
        # -- -- -- For each prediction
        # -- -- -- -- If pred meets match criteria and not already matched, add to matches.
        # -- -- -- For all the matches matches, pick the one with highest score.
        for sample_token in gt_boxes.sample_tokens:
                sample_pred_list = pred_boxes[sample_token]
                sample_gt_list = gt_boxes[sample_token]
                taken = set()  # Initially no gt bounding box is matched.
                
                # check if there are phantom predictions
                class_pred_len = [len([1 for pred in sample_pred_list if pred.detection_name == class_name]) for class_name in conf_mat_mapping]
                class_gt_len = [len([1 for gt in sample_gt_list if gt.detection_name == class_name]) for class_name in conf_mat_mapping]
                
                for i in range(len(class_pred_len)):
                        if class_pred_len[i] > class_gt_len[i]:
                                if list(conf_mat_mapping.keys())[i] == "pedestrian":
                                        x = PED
                                else:
                                        x = OBS
                                # WARNING MIGHT BE OFF
                                # print(f"WARNING: line 40 in distance Param conf")
                                # distance_param_conf_mat[EMPTY][x] += class_pred_len[i] - class_gt_len[i]
                
                for gt in sample_gt_list:
                        
                        best_iou = -1       # Initialize best iou for a bbox with a value that cannot be achieved.
                        best_match = None   # Initialize best matching bbox with None. Tuple of (gt, pred, iou)
                        match_pred_ids = [] # Initialize list of matched predictions for this gt.
                        
                        for i, pred in enumerate(sample_pred_list):
                                if center_distance(pred, gt) < dist_thresh and yaw_diff(pred, gt) < yaw_thresh and i not in taken:
                                        match_pred_ids.append(i)
                                        
                        for match_idx in match_pred_ids:
                                iou = scale_iou(sample_pred_list[match_idx], gt)
                                if best_iou < iou:
                                        best_iou = iou
                                        best_match = (gt, sample_pred_list[match_idx], match_idx)
                        
                        if len(match_pred_ids) == 0:
                                distance_param_conf_mat[conf_mat_mapping[gt.detection_name]][EMPTY] += 1
                                continue
                        else:
                                taken.add(best_match[2])
                                distance_param_conf_mat[conf_mat_mapping[best_match[0].detection_name]][conf_mat_mapping[best_match[1].detection_name]] += 1
                                
                c += 1
                # print(len(sample_pred_list))
                if(sample_token in list_of_validation_tokens):
                        render_sample_data_with_predictions(nusc.get('sample', sample_token)['data']['LIDAR_TOP'], sample_pred_list, nusc=nusc)
                
        assert c == 81