import os
import numpy as np
import traceback
from collections.abc import Iterable
from typing import Tuple, Dict, Any, List
from itertools import chain, combinations
from classes import class_names
from pyquaternion import Quaternion
import datetime
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from utils import EvalBox_ID
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes_render import evalbox_ego_frame, render_sample_data_with_predictions, render_specific_gt_and_predictions, convert_ego_pose_to_flat_veh_coords
from cluster_devel import RadiusBand, Cluster
from pdb import set_trace as st
import pickle as pkl
import sys

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
                 max_dist: int = 100,
                 distance_bin:float = 10,
                 max_dist_bw_obj: float = 2.5,
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
        self.distance_bin = distance_bin
        self.max_dist = max_dist
        self.num_bins = int(max_dist // distance_bin)
        self.radius_bands = []
        self.list_of_classes = list_of_classes
        self.list_of_propositions = None
        self.verbose = verbose
        self.max_dist_bw_obj = max_dist_bw_obj
        self.conf_mat_mapping = conf_mat_mapping
        self.debug = False # Default
        self.class_cm: Dict(Tuple[int, int], np.ndarray) = {}
        self.prop_cm: Dict(Tuple[int, int], np.ndarray) = {}
        self.prop_segmented_cm: Dict(Tuple[int, int], np.ndarray) = {}
        self.disc_gt_boxes: Dict(Tuple[int, int], EvalBoxes) = {}
        self.disc_pred_boxes: Dict(Tuple[int, int], EvalBoxes) = {}
        self.ego_centric_gt_boxes: Dict(Tuple[int, int], EvalBoxes) = {}
        
        self.gt_clusters:dict(Tuple[int, int], RadiusBand) = {}   # {distance_bin: {sample_token: [Cluster1, Cluster2, ...]}

        self.sample_tokens = None

        if self.list_of_classes is not None:
            self.set_list_of_classes(self.list_of_classes)
            self.set_list_of_propositions()
            self.add_empty_label(self.list_of_classes, self.class_dict)
            self.add_empty_label(self.list_of_propositions, self.prop_dict)

        if self.verbose:
            print("Initializing the generator")
        self.initialize()

        if self.verbose:
            print("Loading ground truth and prediction boxes")
        self.__load_boxes()

        if self.verbose:
            print("Loading ground truth and prediction boxes")
        self.group_boxes_into_bands() # 
        self.match_boxes()
        self.initialize_clusters()
        if self.verbose:
            print("Matching boxes")
        
        if self.verbose:
            print("Match boxes completed.")
        # self.__check_distance_param_settings()
        
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)
            
        ##### For debugging purposes #####
        self.list_of_mismatches = []
    
    def __load_ego_veh(self, sample_token:str):
        sample = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        return self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        
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
        if self.verbose:
            print("Results dir:", self.result_path)
            print("Load boxes: No. of Pred tokens:", len(self.pred_boxes.sample_tokens))
            print("Load boxes: No. of GT tokens:", len(self.gt_boxes.sample_tokens))
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
        self.sample_tokens = self.gt_boxes.sample_tokens
        
        if self.verbose:
            print("Completed filtering boxes")

        # Create prediction boxes with ID:
        self.pred_boxes_ID = EvalBoxes()
        for ID, pred_box in enumerate(self.pred_boxes.all):
            eval_box_ID = EvalBox_ID(pred_box, ID)
            self.pred_boxes_ID.add_boxes(sample_token=pred_box.sample_token, boxes=[eval_box_ID])

        for i, gt in enumerate(self.gt_boxes.all):
            self.matches[i] = dict()
            self.matches[i]["GT"] = gt
            self.matches[i]["Pred"] = None

            sample = self.nusc.get('sample', gt.sample_token)
            sample_data_token = sample["data"]["LIDAR_TOP"]
            self.matches[i]["Ego_GT"] = evalbox_ego_frame(sample_data_token, gt, self.nusc, box_type="GT")
            

    def get_gt_idx(self, gt, ego_frame=False):
        '''
        Finds the gt match.
        '''
        gt_tag = "GT"
        if ego_frame:
            gt_tag = "Ego_GT"
            
        for i in self.matches.keys():
            checks = [gt.sample_token == self.matches[i][gt_tag].sample_token]
            checks.append(self.matches[i][gt_tag].translation == gt.translation)
            checks.append(self.matches[i][gt_tag].size == gt.size)
            checks.append(self.matches[i][gt_tag].rotation == gt.rotation)
            checks.append(self.matches[i][gt_tag].detection_name == gt.detection_name)
            # skipping velocity check since it does not matter.
            # checks.append(all(self.matches[i]["GT"].velocity == gt.velocity))
            if gt_tag =="GT":
                checks.append(self.matches[i][gt_tag].ego_translation == gt.ego_translation)
                checks.append(self.matches[i][gt_tag].num_pts == gt.num_pts)
                checks.append(self.matches[i][gt_tag].detection_score == gt.detection_score)
                checks.append(self.matches[i][gt_tag].attribute_name == gt.attribute_name)
            
            try:
                if all(checks):
                    return i
            except:
                st()
        
        print("Error! No ground truth index found.")
        st()

    def match_boxes(self, dist_thresh: float = 2.0, yaw_thresh: float = np.pi/2.0):
        '''
        Match boxes in global coordinate frame
        '''
        for key in self.radius_bands:
            for sample_token in self.sample_tokens:
                preds_in_sample = self.disc_pred_boxes[key][sample_token]
                gt_in_sample = self.disc_gt_boxes[key][sample_token]
                taken = set() # Initially no pred bounding box is matched.
                
                for gt in gt_in_sample:  
                    gt_idx = self.get_gt_idx(gt)  
                    best_iou = -1       # Initialize best iou for a bbox with a value that cannot be achieved.
                    best_match = None   # Initialize best matching bbox with None. Tuple of (pred, gt, iou)
                    match_preds = [] # Initialize list of matched predictions for this gt.

                    for i, pred in enumerate(preds_in_sample):
                        if center_distance(pred, gt) < dist_thresh and yaw_diff(pred, gt) < yaw_thresh and i not in taken:
                            match_preds.append(pred)
            
                    for match in match_preds:
                        iou = scale_iou(match, gt)
                        # If new iou is higher, replace.
                        if iou > best_iou:
                            best_iou = iou
                            best_match = (match, gt)
                    
                    if len(match_preds) > 0:
                        taken.add(i)
                        self.matches[gt_idx]["Pred"] = best_match[0]
        
    def set_debug(self, debug:bool):
        self.debug = debug

    def load_ego_centric_boxes(self) -> None:
        # TODO: Review this function and make consistent with using self.radius_bands in place of dist_bin

        for sample_token in self.sample_tokens:
            sample = self.nusc.get('sample', sample_token)
            _, boxes, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'],
                                               box_vis_level=BoxVisibility.ANY,
                                               use_flat_vehicle_coordinates=True)
            for box in boxes:
                xy_translation = np.array(box.center[:2])
                distance = np.linalg.norm(xy_translation)
                radius_band_idx = np.floor((distance / self.distance_bin))
                
                # TODO handle case when distance is greater than max_dist. Currently ignoring
                if distance > self.max_dist: continue
                radius_band_idx = int(np.floor((distance / self.distance_bin)))
                radius_band = list(self.ego_centric_gt_boxes.keys())[radius_band_idx]
                self.ego_centric_gt_boxes[radius_band][sample_token].append((box))

    def get_distance_to_ego(self, box:EvalBox):
        return np.linalg.norm(box.ego_translation[:2]) # distance to ego in xy

    def get_radius_band(self, box):
        # TODO handle case when distance is greater than max_dist. Currently ignoring
        distance_to_ego = self.get_distance_to_ego(box)
        # if distance > self.max_dist: continue
        radius_band_idx = int(np.floor((distance_to_ego / self.distance_bin)))
        radius_band = self.radius_bands[radius_band_idx]
        return radius_band

    def load_ego_centric_boxes_v2(self) -> None:
        # TODO: Review this function and make consistent with using self.radius_bands in place of dist_bin
        for i, box in enumerate(self.gt_boxes.all):
            sample_token = box.sample_token
            radius_band = self.get_radius_band(box)
            self.ego_centric_gt_boxes[radius_band][sample_token].append(self.matches[i]["Ego_GT"])

    def initialize(self) -> None:
        """ initializes all class variables to their default values
            Have the option of running this function in main once propositions have been set.
        
        Args:
            None
        """
        
        n_class = len(self.list_of_classes)
        n_prop = len(self.list_of_propositions)

        # initializing all the bins
        for i in range(self.num_bins):
            zl = (self.distance_bin * i) + 1
            zu = self.distance_bin * (i + 1)

            self.disc_gt_boxes[(zl, zu)] =  EvalBoxes()
            self.disc_pred_boxes[(zl, zu)] =  EvalBoxes()
            self.class_cm[(zl, zu)] = np.zeros((n_class+1, n_class+1))
            self.prop_cm[(zl, zu)] = np.zeros((n_prop+1, n_prop+1))
            self.prop_segmented_cm[(zl, zu)] = np.zeros((n_prop+1, n_prop+1))
            self.radius_bands.append((zl,zu))
            self.matches = dict()
    
    def group_boxes_into_bands(self):
        # Segmenting the ground truth and prediction boxes into distance bins
        for gt in self.gt_boxes.all:
            gt.ego_translation = (gt.ego_translation[0], gt.ego_translation[1], 0)                         #TODO check if this is working as expected
            dist = np.sqrt(np.dot(gt.ego_translation, gt.ego_translation))
            key = list(self.disc_gt_boxes.keys())[int(dist // self.distance_bin)]      
            self.disc_gt_boxes[key].add_boxes(sample_token=gt.sample_token, boxes=[gt])
            
        for pred in self.pred_boxes.all:
            pred.ego_translation = (pred.ego_translation[0], pred.ego_translation[1], 0)                   #TODO check if this is working as expected
            dist = np.sqrt(np.dot(pred.ego_translation, pred.ego_translation))
            key = list(self.disc_pred_boxes.keys())[int(dist // self.distance_bin)]     
            self.disc_pred_boxes[key].add_boxes(sample_token=pred.sample_token, boxes=[pred])


    def initialize_clusters(self):
        """generates clusters for the ground truth boxes
        
        Hierarchy is as follows:
        - For each distance bin (min radius, max radius) as the dict key
            - For each sample token as the dict key
                - Store a RadiusBand Object
                    - RadiusBand Object contains a list of Cluster objects
                        - Each Cluster object contains a list of ground truth boxes for (theta1 + sigma, theta2)
        """
        for band in self.radius_bands:
            self.gt_clusters[band] = {}  
            self.ego_centric_gt_boxes[band] = {}           
            for sample_token in self.sample_tokens:
                self.ego_centric_gt_boxes[band][sample_token] = []        
            
        self.load_ego_centric_boxes_v2() # Populating self.ego_centric_gt_boxes
        
        for band in self.radius_bands:
            for sample_token in self.sample_tokens:
                self.gt_clusters[band][sample_token] = \
                    RadiusBand(sample_token = sample_token, 
                                ego_veh=self.__load_ego_veh(sample_token),
                                gt_boxes = self.ego_centric_gt_boxes[band][sample_token],
                                max_distance_bw_obj = self.max_dist_bw_obj, 
                                radius_band=band)
            
    def __check_distance_param_settings(self) -> None:
        """
        Check that the distance parametrization settings are valid.
        """
        if self.distance_parametrized:
            assert self.lower_thresh < self.upper_thresh, 'Error: lower_thresh must be lesser than upper_thresh'
            assert self.distance_bin > 0, 'Error: distance_bin must be > 0'
            

    ### ------- Propositions ----------- ####
    def get_propositions(self):
        n = len(self.list_of_classes)
        propositions = list(powerset(self.list_of_classes))
        self.prop_dict = dict()
        
        for k, prop in enumerate(propositions):
            if any(prop): # if not empty
                prop_label = set(prop)
                self.prop_dict[k-1] = prop_label
            else:
                self.prop_dict[len(propositions)-1] = set(["empty"])

    def set_list_of_propositions(self):
        self.get_propositions()
        n = len(self.prop_dict)
        self.list_of_propositions = list(self.prop_dict.values())

    def set_list_of_classes(self, list_of_classes):
        self.list_of_classes = list_of_classes
        self.class_dict = {k:c for k,c in enumerate(list_of_classes)}
    
    def add_empty_label(self, label_list, label_dict):
        if type(label_list[0]) == str:
            empty_elem = "empty"    
        else:
            empty_elem = set(["empty"])
        if empty_elem not in label_list and empty_elem not in list(label_dict.items()):
            label_list.append(empty_elem)
            kempty = len(label_dict)
            label_dict.update({kempty:empty_elem})

    def get_list_of_classes(self):
        return self.list_of_classes, self.class_dict

    def get_list_of_propositions(self):
        return self.list_of_propositions, self.prop_dict
        
    def custom_propositions(self):
        # Class to set custom propositions.
        pass

    ### ------- End of Propositions ----------- ####

    ### ------- Code to compute class labeled confusion matrices ----------- ####

    def get_class_cm(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Get a dictionary with the distance parametrized confusion matrices for each distance bin.
        Args:
            None
        Returns:
            A dictionary where the keys are tuples of the form (lower_dist_thresh, upper_dist_thresh)
            The values are the corresponding distance parameterized confusion matrices. 
        """
        n = len(self.list_of_classes)
        for key in list(self.disc_gt_boxes.keys()):
            self.class_cm[key] = np.zeros((n,n))
            # self.class_cm[key] = self.compute_class_labeled_cm(self.disc_gt_boxes[key], self.disc_pred_boxes[key], self.conf_mat_mapping)
            for sample_token in self.sample_tokens:
                for gt_box in self.disc_gt_boxes[key][sample_token]:
                    pred_box = self.get_matched_pred(gt_box)
                    evaluation = self.single_evaluation_class_cm(gt_box, pred_box)
                    self.class_cm[key] += evaluation
        return self.class_cm
    
    def get_labels_for_boxes(self, boxes):
        classes = set()
        for box in boxes:
            if box:
                classes.add(box.detection_name) 
                if box.detection_name not in {"pedestrian", "car", "truck", "bus", "traffic_cone", "bicycle", "construction_vehicle", "barrier", "motorcycle","trailer"}:
                    st()
        classes = [c for c in classes if c in {"pedestrian", "car", "truck", "bus"}]
        classes = set({"ped" if x == "pedestrian" else "obs" for x in classes})
        if len(classes) == 0:
            classes = set({"empty"})
        return classes

    def get_class_cm_indices(self, gt_box, pred_box):
        try:
            gt_label = self.get_labels_for_boxes([gt_box]).pop()
            pred_label = self.get_labels_for_boxes([pred_box]).pop()
        except:
            traceback.print_exc()
        
        gt_idx = None
        pred_idx = None
        
        for k, class_label in self.class_dict.items():
            if gt_label == class_label:
                gt_idx = k
            if pred_label == class_label:
                pred_idx = k
        return gt_idx, pred_idx

    def single_evaluation_class_cm(self, gt_box:EvalBoxes, pred_box: EvalBoxes) -> np.ndarray:
        # single evaluation for proposition labeled confusion matrix
        n = len(self.list_of_classes)
        class_cm = np.zeros((n,n))   
        gt_idx, pred_idx = self.get_class_cm_indices(gt_box, pred_box)
        if (gt_idx is not None) and (pred_idx is not None):
            class_cm[pred_idx][gt_idx] += 1   
        if (gt_idx is None and pred_idx is not None) or (pred_idx is None and gt_idx is not None):
            print("Error: One of the confusion matrix indices is returned as None. Check")
        return class_cm

    ### ------- End of Code to compute class labeled confusion matrices ----------- ####

    ### ------- Code to compute proposition labeled confusion matrices ----------- ####
    def get_matched_pred(self, gt,ego_frame=False):
        gt_idx = self.get_gt_idx(gt, ego_frame=ego_frame)
        pred_box = self.matches[gt_idx]["Pred"]
        return pred_box

    def get_prop_cm(self):
        """Get a dictionary with the proposition labelled confusion matrices for each distance bin.
        Args:
            None
            
        Returns:
            A dictionary where the keys are tuples of the form (lower_dist_thresh, upper_dist_thresh)
            The values are the corresponding proposition labelled confusion matrices.
        """
        n = len(self.list_of_propositions)

        # Debugging figures stored here:
        if self.debug:
            self.plot_folder = os.path.join("plots/prop_cm_debug_plots",datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            if not os.path.exists(self.plot_folder):
                os.makedirs(self.plot_folder)
            self.mismatched_samples = []

        # Looping over radius bands
        for key in list(self.disc_gt_boxes.keys()):
            self.prop_cm[key] = np.zeros((n,n))

            # Loop over samples:
            for sample_token in self.sample_tokens:
                gt_boxes = self.disc_gt_boxes[key][sample_token].copy()
                matched_pred_boxes = [self.get_matched_pred(gt) for gt in gt_boxes]
                # evaluation = self.single_evaluation_prop_cm(gt_boxes, self.disc_pred_boxes[key][sample_token])
                evaluation = self.single_evaluation_prop_cm(gt_boxes, matched_pred_boxes)
                self.prop_cm[key] += evaluation

        if self.debug:
            mismatched_samples_pkl = f"{self.plot_folder}/mismatched_sample_tokens.pkl"
            with open(mismatched_samples_pkl, "wb") as f:
                pkl.dump(self.mismatched_samples, f)
            f.close() 
        return self.prop_cm
    
    def get_prop_cm_indices(self, gt_boxes, pred_boxes, ref_frame="global"):
        """
        Returns the predicted and ground truth indices of a confusion
        matrix for the given set of gt_classes and pred_classes
        gt_boxes: List[Box]
        pred_boxes: List[Box]
        """
        
        try:
            gt_labels = self.get_labels_for_boxes(gt_boxes)
            pred_labels = self.get_labels_for_boxes(pred_boxes)
        except:
            traceback.print_exc()
        
        if gt_labels != pred_labels and self.debug:
            if len(gt_boxes) > 0: self.list_of_mismatches.append(gt_boxes[0].sample_token)
            self.render_predictions(gt_boxes, pred_boxes, ref_frame=ref_frame)

        gt_idx = None
        pred_idx = None
        if gt_labels == set({"empty"}) and pred_labels == set({"empty"}):
            return gt_idx, pred_idx
        for k, prop_label in self.prop_dict.items():
            if gt_labels == prop_label:
                gt_idx = k
            if pred_labels == prop_label:
                pred_idx = k
        return gt_idx, pred_idx

    def single_evaluation_prop_cm(self, gt_boxes:EvalBoxes, pred_boxes: EvalBoxes, ref_frame="global") -> np.ndarray:
        # single evaluation for proposition labeled confusion matrix
        n = len(self.list_of_propositions)
        prop_cm = np.zeros((n,n))   
        gt_idx, pred_idx = self.get_prop_cm_indices(gt_boxes, pred_boxes, ref_frame=ref_frame)
        if (gt_idx is not None) and (pred_idx is not None):
            prop_cm[pred_idx][gt_idx] += 1   
        if (gt_idx is None and pred_idx is not None) or (pred_idx is None and gt_idx is not None):
            print("Error: One of the confusion matrix indices is returned as None. Check")
        return prop_cm

    ### ------- End to compute proposition labeled confusion matrices ----------- ####

    ### ------- Code to compute clustered evaluated proposition labeled confusion matrices ----------- ####

    def get_prop_segmented_cm(self):   
        n = len(self.list_of_propositions)
        # Debugging figures stored here:
        if self.debug:
            self.plot_folder = os.path.join("plots/clustered_prop_cm_debug_plots",datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            if not os.path.exists(self.plot_folder):
                os.makedirs(self.plot_folder)
            
            self.mismatched_samples = []

        for radius_band in list(self.gt_clusters.keys()): # -> loop through distance params
            self.prop_segmented_cm[radius_band] = np.zeros((n,n))
            # Loop over samples:
            for sample_token in self.sample_tokens:
                # Radius band object. 
                gt_boxes = self.disc_gt_boxes[radius_band][sample_token].copy()
                radius_band_obj = self.gt_clusters[radius_band][sample_token]
                for cluster in radius_band_obj.clusters:
                    # pred_boxes_in_cluster = self.find_preds_for_cluster(cluster, dist_thresh=2.0)
                    matched_pred_boxes = [self.get_matched_pred(gt,ego_frame=True) for gt in cluster.boxes]
                
                    evaluation = self.single_evaluation_prop_cm(cluster.boxes, matched_pred_boxes, ref_frame="ego") # in radians
                    self.prop_segmented_cm[radius_band] += evaluation
        
        if self.debug:
            mismatched_samples_pkl = f"{self.plot_folder}/mismatched_sample_tokens.pkl"
            with open(mismatched_samples_pkl, "wb") as f:
                pkl.dump(self.mismatched_samples, f)
            f.close() 
        
        return self.prop_segmented_cm       
    
    ### ------- End to compute clustered evaluated proposition labeled confusion matrices ----------- ####
    def render_predictions(self, gt_boxes, pred_boxes, ref_frame="global"):
        if gt_boxes == []:
            assert pred_boxes != []
            sample_token = pred_boxes[0].sample_token
        else:
            sample_token = gt_boxes[0].sample_token

        if sample_token not in self.mismatched_samples:
            self.mismatched_samples.append(sample_token)

        gt_info = []
        pred_info = []
        
        for box in gt_boxes:
            evalbox_to_box = convert_from_EvalBox_to_Box(box)
            [label] = self.get_labels_for_boxes([box])
            gt_info.append((evalbox_to_box, "gt: " + label))
        
        pred_boxes = [box for box in pred_boxes if box is not None] # Filtering out None

        for box in pred_boxes:
            evalbox_to_box = convert_from_EvalBox_to_Box(box)
            [label] = self.get_labels_for_boxes([box])
            pred_info.append((evalbox_to_box, "pred: " + label))
        render_specific_gt_and_predictions(sample_token, gt_info, pred_info, self.nusc, self.plot_folder,ref_frame=ref_frame)


#### --------- Utils functions --------- #####
def powerset(iterable: Iterable):
    """powerset function to generate all possible subsets of any iterable

    Args:
        iterable (Iterable): The iterable to create the powerset of

    Returns:
        An iterable chain object containing all possible subsets of the input iterable
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1)) 



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
    
def convert_from_Box_to_EvalBox(box:Box) -> EvalBox:
    """Converts a Box object to an EvalBox object
    """
    
    # print(f"*******Rotation of an Box***| {box.orientation.elements.tolist()} |***********")
    # print(f"-------> Velocity of an Box ---> {box.velocity} <---------")
    
    return DetectionBox(
        translation = box.center.tolist(),
        size = box.wlh.tolist(),
        rotation = box.orientation.elements.tolist(),
        velocity = (np.nan, np.nan),
        sample_token=box.token,
        detection_name=convert_specificLabel_to_genericLabel(box.name)
    )
    
def convert_specificLabel_to_genericLabel(label:str) -> str:
    """Converts a specific label to a generic label
    """
    if label in {
        "human.pedestrian.adult",
        "human.pedestrian.child",
        'human.pedestrian.construction_worker',
        "human.pedestrian.personal_mobility",
        "human.pedestrian.police_officer",
        "human.pedestrian.stroller",
        "human.pedestrian.wheelchair",
    }:
        return "pedestrian"

    if label in {
        "movable_object.barrier",
        "movable_object.debris",
        "movable_object.pushable_pullable",
        "static_object.bicycle_rack",
    }:
        return "barrier"

    if label == "movable_object.trafficcone":
        return "traffic_cone"

    if label == "vehicle.bicycle":
        return "bicycle"

    if label in {"vehicle.bus.bendy", "vehicle.bus.rigid"}:
        return "bus"

    if label in {"vehicle.car", "vehicle.emergency.police"}:
        return "car"

    if label == "vehicle.motorcycle":
        return "motorcycle"

    if label in {"vehicle.truck", "vehicle.emergency.ambulance"}:
        return "truck"

    if label == "vehicle.construction":
        return "construction_vehicle"

    if label == "vehicle.trailer":
        return "trailer"
    
    raise ValueError(f"GenConfMat/label:618   Error: label {label} not found in the list of classes")