# Script to generate confusion matrix for class labels

# Copy of the evaluate.py script from nuScenes
import os
import numpy as np
from typing import Tuple, Dict, Any, List
from collections.abc import Iterable
from itertools import chain, combinations

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from main_nuscenes_render import render_sample_data_with_predictions, render_specific_gt_and_predictions

from pdb import set_trace as st
from collections import OrderedDict as od

from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from pathlib import Path

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
                 ) -> None:
        """Initialize a DetectionEval object.
            
            Args:
                nusc: A NuScenes object.
                config: A DetectionConfig object.
                result_path: Path of the nuScenes JSON result file.
                eval_set: The dataset split to evaluate on, e.g. train, val or test.
                output_dir: Folder to save plots and results to.
                verbose: Whether to print to stdout.
                distance_parametrized: Whether the confusion matrix is parametrized by distance or not.
                lower_thresh: Lower distance threshold.
                upper_thresh: Upper distance threshold.
                max_dist: Maximum distance to consider for the distance parametrized confusion matrix.
                distance_bin: If lower_thresh = -1 and upper_thresh = inf, there is only one confusion matrix.
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
        self.classes = list_of_classes
        self.verbose = verbose
        self.dist_conf_mats: dict(Tuple[int, int], np.ndarray) = {}
        self.prop_conf_mats: dict(Tuple[int, int], np.ndarray) = {}
        self.disc_gt_boxes: dict(Tuple[int, int], EvalBoxes) = {}
        self.disc_pred_boxes: dict(Tuple[int, int], EvalBoxes) = {}
        self.conf_mat_mapping = conf_mat_mapping
        
        self.__load_boxes()
        self.__initialize()
        # self.__check_distance_param_settings()
        
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)
        
        self.sample_tokens = self.gt_boxes.sample_tokens


    def __initialize(self) -> None:
        """ initializes all class variables to their default values
        
        Args:
            None
        """
        
        n = len(self.classes)
        
        # initializing all the bins
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

        # Segmenting the ground truth and prediction boxes into distance bins
        for gt in self.gt_boxes.all:
            dist = np.sqrt(np.dot(gt.ego_translation, gt.ego_translation))
            key = list(self.disc_gt_boxes.keys())[int(dist // self.distance_bin)]      # TODO check if this is correct at the edges
            self.disc_gt_boxes[key].add_boxes(sample_token=gt.sample_token, boxes=[gt])
            
        for pred in self.pred_boxes.all:
            dist = np.sqrt(np.dot(pred.ego_translation, pred.ego_translation))
            key = list(self.disc_pred_boxes.keys())[int(dist // self.distance_bin)]      # TODO check if this is correct at the edges
            self.disc_pred_boxes[key].add_boxes(sample_token=pred.sample_token, boxes=[pred])

    def __load_boxes(self) -> None:
        """Loads GT annotations and predictions from respective files and saves them in respective class variables.
        Args: 
            None
        """
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

    def __check_distance_param_settings(self) -> None:
        """
            Check that the distance parametrization settings are valid.
        """
        if self.distance_parametrized:
            assert self.lower_thresh < self.upper_thresh, 'Error: lower_thresh must be lesser than upper_thresh'
            assert self.distance_bin > 0, 'Error: distance_bin must be > 0'
            
            
    def get_distance_param_conf_mat(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Get a dictionary with the distance parametrized confusion matrices for each distance bin.
        
        Args:
            None
        
        Returns:
            A dictionary where the keys are tuples of the form (lower_dist_thresh, upper_dist_thresh)
            The values are the corresponding distance parameterized confusion matrices. 
        """
        for key in list(self.disc_gt_boxes.keys()):
            self.dist_conf_mats[key] = self.calculate_conf_mat(self.disc_gt_boxes[key], self.disc_pred_boxes[key], self.conf_mat_mapping)

        return self.dist_conf_mats
    
    
    def get_proposition_labelled_conf_mat(self):
        """Get a dictionary with the proposition labelled confusion matrices for each distance bin.
        
        Args:
            None
            
        Returns:
            A dictionary where the keys are tuples of the form (lower_dist_thresh, upper_dist_thresh)
            The values are the corresponding proposition labelled confusion matrices.
        """
        for key in list(self.disc_gt_boxes.keys()):
            # debug_sample_token = "0bb62a68055249e381b039bf54b0ccf8"
            # if sample_token == debug_sample_token:
            #     print("Radius band: ", key)
            self.prop_conf_mats[key], prop_dict = self.compute_prop_cm(self.disc_gt_boxes[key], self.disc_pred_boxes[key], ["ped", "obs"], self.classes)
    
        return self.prop_conf_mats, prop_dict
    

    def calculate_conf_mat(self,
                            gt_boxes:EvalBoxes, 
                            pred_boxes: EvalBoxes, 
                            conf_mat_mapping: Dict,
                            dist_thresh: float = 2.0,       # in m 
                            yaw_thresh: float = np.pi/2.0): # in radians  -> np.ndarray:

        EMPTY = len(self.classes)
        distance_param_conf_mat = np.zeros( (len(self.classes)+1, len(self.classes)+1) )
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
                
                

                for gt in sample_gt_list:
                    best_iou = -1       # Initialize best iou for a bbox with a value that cannot be achieved.
                    best_match = None   # Initialize best matching bbox with None. Tuple of (gt, pred, iou)
                    match_pred_ids = [] # Initialize list of matched predictions for this gt.
                    
                    for i, pred in enumerate(sample_pred_list):
                            st()
                            if center_distance(pred, gt) < dist_thresh and yaw_diff(pred, gt) < yaw_thresh and i not in taken:
                                    match_pred_ids.append(i)
                                    
                    for match_idx in match_pred_ids:
                            iou = scale_iou(sample_pred_list[match_idx], gt)
                            if best_iou < iou:
                                    best_iou = iou
                                    best_match = (sample_pred_list[match_idx], gt, match_idx)

                    if len(match_pred_ids) == 0:
                            distance_param_conf_mat[EMPTY][conf_mat_mapping[gt.detection_name]] += 1
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
    
    def powerset(self, iterable: Iterable):
        """powerset function to generate all possible subsets of any iterable

        Args:
            iterable (Iterable): The iterable to create the powerset of

        Returns:
            An iterable chain object containing all possible subsets of the input iterable
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))   
    
    def __unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def __angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.__unit_vector(v1)
        v2_u = self.__unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def cluster(self,
                gt_boxes:EvalBoxes, 
                pred_boxes: list, 
                list_of_classes: list) -> np.ndarray:
        
        n = len(self.classes)
        gt_vectors:Dict(EvalBox, np.ndarray) = {}
        pred_vectors:Dict(EvalBox, np.ndarray) = {}
        # Make new Eval Boxes object
        # Iterate through each sample token
        # For each sample token, iterate through each ground truth box
        # Caclulate the vector between the ego vehicle - ground truth box and ego - prediction box
        # Add it to the vectors dictionary
        
        for sample_token in gt_boxes.sample_tokens:
            # get the orientation of the ego vehicle
                # -- Orientation of the vehicle is given as a quaternion
                # -- The vector, calculated with ](x1-x2), (y1-y2), ...] etc is Euler I believe?
            sample = self.nusc.get('sample', sample_token)
            sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            self.nusc.get('ego_pose', sd_record['ego_pose_token'])
            
            sample_pred_list = pred_boxes[sample_token]
            sample_gt_list = gt_boxes[sample_token]
            
            something = "ego's orientation that is the same for everyting in this sample" 
            
            for gt in sample_gt_list:
                gt_vectors[gt] = self.__angle_between(gt.ego_translation, something)

    def get_propositions(self):
        n = len(self.classes)
        propositions = list(self.powerset(self.classes))
        prop_dict = dict()
        for k, prop in enumerate(propositions):
            if any(prop): # if not empty
                prop_label = set(prop)
                prop_dict[k] = prop_label
            else:
                prop_dict[k] = set(["empty"])
        return prop_dict

    def compute_prop_cm(self, gt_boxes:EvalBoxes, pred_boxes: list, list_of_propositions: list, 
                        class_names:list) -> np.ndarray:
        # Comments: list_of_propositions, self.classes, and class_names are the exact same.
        # Pass in propositions
        prop_dict = self.get_propositions()
        n = len(prop_dict)
        prop_cm = np.zeros((n,n))        

        for sample_token in gt_boxes.sample_tokens:
            sample_pred_list = pred_boxes[sample_token]
            sample_gt_list = gt_boxes[sample_token]
            taken = set()  # Initially no gt bounding box is matched.

            gt_classes = {gt.detection_name for gt in sample_gt_list}
            pred_classes = {pred.detection_name for pred in sample_pred_list}

            #TODO convert into generic
            gt_classes = set({"ped" if x == "pedestrian" else "obs" for x in gt_classes})
            pred_classes = set({"ped" if x == "pedestrian" else "obs" for x in pred_classes})

            if len(gt_classes) == 0:
                gt_classes = set({"empty"})
            if len(pred_classes) == 0:
                pred_classes = set({"empty"})

            
            gt_idx = 0
            pred_idx = 0

            for k, prop_label in prop_dict.items():
                if gt_classes == prop_label:
                    gt_idx = k
                if pred_classes == prop_label:
                    pred_idx = k
                
            prop_cm[pred_idx][gt_idx] += 1
            
        return prop_cm, prop_dict
    
    # ====================== Functions not in main that do not affect computations ======================= #
    def get_labels_for_boxes(self, boxes):
        classes = {box.detection_name for box in boxes}
        classes = set({"ped" if x == "pedestrian" else "obs" for x in classes})
        if len(classes) == 0:
            classes = set({"empty"})
        return classes

    def render_predictions(self, mismatched_samples, plot_folder):
        '''
        Render predictions for debugging to compare with latest code
        '''
        for sample_token in mismatched_samples:
            st()
            gt_boxes = [box for box in self.gt_boxes.all if box.sample_token == sample_token]
            pred_boxes = [box for box in self.pred_boxes.all if box.sample_token == sample_token]
        
        gt_info = []
        pred_info = []
        Box_gt_boxes = []
        Box_pred_boxes = []

        for box in gt_boxes:
            evalbox_to_box = convert_from_EvalBox_to_Box(box)
            [label] = self.get_labels_for_boxes([box])
            gt_info.append((evalbox_to_box, "gt: " + label))
            Box_gt_boxes.append(evalbox_to_box)
        
        for box in pred_boxes:
            evalbox_to_box = convert_from_EvalBox_to_Box(box)
            [label] = self.get_labels_for_boxes([box])
            pred_info.append((evalbox_to_box, "pred: " + label))
            Box_pred_boxes.append(evalbox_to_box)

        out_path = Path(f"{plot_folder}/sample_token_{sample_token}/main_mismatch.png")
        # render_sample_data_with_predictions(sample_token, out_path=out_path, nusc=self.nusc, pred_boxes=Box_pred_boxes)
        render_specific_gt_and_predictions(sample_token, gt_info, pred_info, nusc=self.nusc, out_path=out_path)

def convert_from_EvalBox_to_Box(eval_box:EvalBox) -> Box:
    """Converts an EvalBox object to a Box object
    """
    # print(f"Rotation of an EvalBox {(eval_box.rotation)}")
    box = Box(
        center=eval_box.translation,
        size=eval_box.size,
        orientation=Quaternion(eval_box.rotation),
        token=eval_box.sample_token
    )
    
    if type(eval_box) == DetectionBox:
        box.name = eval_box.detection_name
        box.score = eval_box.detection_score
    return box

# ====================== End of Functions not in main that do not affect computations ======================= #
