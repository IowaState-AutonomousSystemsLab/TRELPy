# Script to generate confusion matrix for class labels

# Copy of the evaluate.py script from nuScenes
import os
import numpy as np
from typing import Tuple, Dict, Any, List
from itertools import chain, combinations

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes_render import render_sample_data_with_predictions

class GenerateConfusionMatrix:
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
                 conf_mat_mapping: Dict = None,
                 list_of_classes: List = None,
                 distance_parametrized: bool = False,
                #  lower_thresh: float = -1,
                #  upper_thresh: float = np.inf,
                 max_dist: int = 100,
                 distance_bin:float = 10,
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
            :param max_dist: Maximum distance to consider for the distance parametrized confusion matrix.
            :param distance_bin: If lower_thresh = -1 and upper_thresh = inf, there is only one confusion matrix.
        """
        
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.distance_parametrized = distance_parametrized
        # self.lower_thresh = lower_thresh
        # self.upper_thresh = upper_thresh
        self.distance_bin = distance_bin
        self.max_dist = max_dist
        self.num_bins = int(max_dist // distance_bin)
        self.list_of_classes = list_of_classes
        self.verbose = verbose
        self.dist_conf_mats: dict(Tuple[int, int], np.ndarray) = {}
        self.prop_conf_mats: dict(Tuple[int, int], np.ndarray) = {}
        self.disc_gt_boxes: dict(Tuple[int, int], EvalBoxes) = {}
        self.disc_pred_boxes: dict(Tuple[int, int], EvalBoxes) = {}
        self.conf_mat_mapping = conf_mat_mapping
        
        self.load_boxes()
        self.initialize()
        # self.check_distance_param_settings()
        
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)
        
        _ = None
        
        # for pred in self.pred_boxes.all:
        #     dist = np.sqrt(np.dot(pred.ego_translation, pred.ego_translation)) 
        #     if dist > upper_thresh or dist < lower_thresh:
        #         self.pred_boxes.all.remove(pred)
        #         self.pred_boxes.sample_tokens.remove(pred.sample_token)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def initialize(self) -> None:
        
        n = len(self.list_of_classes)
        for i in range(self.num_bins):
            if i == 0:
                self.disc_gt_boxes[(0, self.distance_bin)] = EvalBoxes()
                self.disc_pred_boxes[(0, self.distance_bin)] = EvalBoxes()
                self.dist_conf_mats[(0, self.distance_bin)] = np.zeros((n+1, n+1))
                self.prop_conf_mats[(0, self.distance_bin)] = np.zeros(((2**n), (2**n)))
            else:
                self.disc_gt_boxes[( (self.distance_bin * i)+1, self.distance_bin * (i + 1) )] = EvalBoxes()
                self.disc_pred_boxes[( (self.distance_bin * i)+1, self.distance_bin * (i + 1) )] = EvalBoxes()
                self.dist_conf_mats[( (self.distance_bin * i)+1, self.distance_bin * (i + 1) )] = np.zeros((n+1, n+1))
                self.prop_conf_mats[( (self.distance_bin * i)+1, self.distance_bin * (i + 1) )] = np.zeros(((2**n), (2**n)))
                        
        for gt in self.gt_boxes.all:
            dist = np.sqrt(np.dot(gt.ego_translation, gt.ego_translation))
            key = list(self.disc_gt_boxes.keys())[int(dist // self.distance_bin)]      # TODO check if this is correct at the edges
            self.disc_gt_boxes[key].add_boxes(sample_token=gt.sample_token, boxes=[gt])
            
        for pred in self.pred_boxes.all:
            dist = np.sqrt(np.dot(pred.ego_translation, pred.ego_translation))
            key = list(self.disc_pred_boxes.keys())[int(dist // self.distance_bin)]      # TODO check if this is correct at the edges
            self.disc_pred_boxes[key].add_boxes(sample_token=pred.sample_token, boxes=[pred])

    def load_boxes(self):
        # Load data.
        if self.verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=self.verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=self.verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if self.verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(self.nusc, self.pred_boxes, self.cfg.class_range, verbose=self.verbose)
        if self.verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(self.nusc, self.gt_boxes, self.cfg.class_range, verbose=self.verbose)
    
    def compute_distance_param_conf_mat(self):
        for key in list(self.disc_gt_boxes.keys()):
            self.dist_conf_mats[key] = self.calculate_conf_mat(self.disc_gt_boxes[key], self.disc_pred_boxes[key], self.conf_mat_mapping)
    
    def get_distance_param_conf_mat(self):
        self.compute_distance_param_conf_mat() # Todo: Check if the matrix has been constructed, and then return the correct one.
        return self.dist_conf_mats
    
    
    def check_distance_param_settings(self) -> None:
        """
            Check that the distance parametrization settings are valid.
        """
        if self.distance_parametrized:
            assert self.lower_thresh < self.upper_thresh, 'Error: lower_thresh must be lesser than upper_thresh'
            assert self.distance_bin > 0, 'Error: distance_bin must be > 0'

    def calculate_conf_mat(self,
                                          gt_boxes:EvalBoxes, 
                                          pred_boxes: EvalBoxes, 
                                          conf_mat_mapping: Dict,
                                          dist_thresh: float = 2.0,       # in m 
                                          yaw_thresh: float = np.pi/2.0): # in radians  -> np.ndarray:

        EMPTY = len(self.list_of_classes)
        distance_param_conf_mat = np.zeros( (len(self.list_of_classes)+1, len(self.list_of_classes)+1) )
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
                
                # for i in range(len(class_pred_len)):
                        # if class_pred_len[i] > class_gt_len[i]:
                                # if list(conf_mat_mapping.keys())[i] == "pedestrian":
                                #         x = PED
                                # else:
                                #         x = OBS
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
                # if self.validation and (sample_token in list_of_validation_tokens):
                #         render_sample_data_with_predictions(self.nusc.get('sample', sample_token)['data']['LIDAR_TOP'], sample_pred_list, nusc=self.nusc)
                
        # assert c == 81
        return distance_param_conf_mat
    
    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))    
    
    def calculate_prop_labelled_conf_mat(self, gt_boxes:EvalBoxes, 
                                      pred_boxes: list, 
                                      list_of_propositions: list, 
                                      class_name:str) -> np.ndarray:
    
        propn_indices = list(range(len(list_of_propositions)))
        propn_powerset = list(powerset(propn_indices))

        for sample_token in gt_boxes.sample_tokens:
            sample_pred_list = pred_boxes[sample_token]
            sample_gt_list = gt_boxes[sample_token]
            taken = set()  # Initially no gt bounding box is matched.

            gt_classes   = set([gt.detection_name for gt in sample_gt_list])
            pred_classes = set([pred.detection_name for pred in sample_gt_list])
            
            gt_classes = {"ped" if x == "pedestrian" else "obs" for x in gt_classes}
            pred_classes = {"ped" if x == "pedestrian" else "obs" for x in pred_classes}

            gt_idx = 0
            gt_propn = "empty"
            pred_idx = 0
            pred_propn = "empty"

            for i, propn in enumerate(propn_powerset):
                
                classes = {} if len(propn) == 0 else {list_of_propositions[c] for c in propn}
                
                if gt_classes == set(classes):
                    gt_propn = set(propn)
                    gt_idx = i
                if pred_classes == set(classes):
                    pred_propn = set(propn)
                    pred_idx = i

            propn_labelled_conf_mat[pred_idx][gt_idx] += 1
        
