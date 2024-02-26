import os
import numpy as np
from collections.abc import Iterable
from typing import Tuple, Dict, Any, List
from itertools import chain, combinations
from classes import class_names

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes_render import convert_EvalBox_to_flat_veh_coords
from cluster_devel import RadiusBand, Cluster
from pdb import set_trace as st

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
                 max_dist_bw_obj: float = 2.0,
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
        
        self.dist_conf_mats: Dict(Tuple[int, int], np.ndarray) = {}
        self.prop_conf_mats: Dict(Tuple[int, int], np.ndarray) = {}
        self.clustered_conf_mats: Dict(Tuple[int, int], np.ndarray) = {}
        self.disc_gt_boxes: Dict(Tuple[int, int], EvalBoxes) = {}
        self.disc_pred_boxes: Dict(Tuple[int, int], EvalBoxes) = {}
        self.ego_centric_gt_boxes: Dict(Tuple[int, int], EvalBoxes) = {}
        
        self.gt_clusters:dict(Tuple[int, int], RadiusBand) = {}   # {distance_bin: {sample_token: [Cluster1, Cluster2, ...]}

        self.sample_tokens = None

        if self.list_of_classes is not None:
            self.set_list_of_classes(self.list_of_classes)
            self.set_list_of_propositions()
        self.initialize()
        self.__load_boxes()
        self.initialize_clusters()
        # st()
        # self.__check_distance_param_settings()
        
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)
    
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
        
    def load_ego_centric_boxes(self) -> None:
        for sample_token in self.sample_tokens:
            sample = self.nusc.get('sample', sample_token)
            _, boxes, _ = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'],
                                               box_vis_level=BoxVisibility.ANY,
                                               use_flat_vehicle_coordinates=True)

            for box in boxes:
                xy_translation = np.array(box.center[:2])
                distance = np.linalg.norm(xy_translation)
                dist_band_idx = np.floor((distance / self.distance_bin))
                
                #TODO handle case when distance is greater than max_dist. Currently ignoring
                if distance > self.max_dist: continue
                dist_band_idx = int(np.floor((distance / self.distance_bin)))
                dist_band = list(self.ego_centric_gt_boxes.keys())[dist_band_idx]
                self.ego_centric_gt_boxes[dist_band][sample_token].append(box)
                
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
            self.dist_conf_mats[(zl, zu)] = np.zeros((n_class+1, n_class+1))
            self.prop_conf_mats[(zl, zu)] = np.zeros((n_prop+1, n_prop+1))
            self.clustered_conf_mats[(zl, zu)] = np.zeros((n_prop+1, n_prop+1))
            self.radius_bands.append((zl,zu))

            # if i == 0:
            #     self.disc_gt_boxes[(0, self.distance_bin)] = EvalBoxes()
            #     self.disc_pred_boxes[(0, self.distance_bin)] = EvalBoxes()
            #     self.dist_conf_mats[(0, self.distance_bin)] = np.zeros((n+1, n+1))
            #     self.prop_conf_mats[(0, self.distance_bin)] = np.zeros(((2**n), (2**n)))
            #     self.prop_conf_mats_w_clustering[(0, self.distance_bin)] = np.zeros(((2**n), (2**n)))
            # else:
            #     self.disc_gt_boxes[( (self.distance_bin * i)+1, self.distance_bin * (i + 1) )] = EvalBoxes()
            #     self.disc_pred_boxes[( (self.distance_bin * i)+1, self.distance_bin * (i + 1) )] = EvalBoxes()
            #     self.dist_conf_mats[( (self.distance_bin * i)+1, self.distance_bin * (i + 1) )] = np.zeros((n+1, n+1))
            #     self.prop_conf_mats[( (self.distance_bin * i)+1, self.distance_bin * (i + 1) )] = np.zeros(((2**n), (2**n)))
    
    def initialize_clusters(self):
        self.generate_clusters()
     
        # Segmenting the ground truth and prediction boxes into distance bins
        for gt in self.gt_boxes.all:
            gt.ego_translation = (gt.ego_translation[0], gt.ego_translation[1], 0)                         #TODO check if this is working as expected
            dist = np.sqrt(np.dot(gt.ego_translation, gt.ego_translation))
            key = list(self.disc_gt_boxes.keys())[int(dist // self.distance_bin)]      
            self.disc_gt_boxes[key].add_boxes(sample_token=gt.sample_token, boxes=[gt])
            
        for pred in self.pred_boxes.all:
            pred.ego_translation = (pred.ego_translation[0], pred.ego_translation[1], 0)                         #TODO check if this is working as expected
            dist = np.sqrt(np.dot(pred.ego_translation, pred.ego_translation))
            key = list(self.disc_pred_boxes.keys())[int(dist // self.distance_bin)]     
            self.disc_pred_boxes[key].add_boxes(sample_token=pred.sample_token, boxes=[pred])


    def generate_clusters(self):
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
            
        self.load_ego_centric_boxes()
        
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
                self.prop_dict[k] = prop_label
            else:
                self.prop_dict[k] = set(["empty"])

    def set_list_of_propositions(self):
        self.get_propositions()
        n = len(self.prop_dict)
        self.list_of_propositions = list(self.prop_dict.values())

    def set_list_of_classes(self, list_of_classes):
        self.list_of_classes = list_of_classes
        self.class_dict = {k:c for k,c in enumerate(list_of_classes)}
        if "empty" not in list_of_classes or "Empty" not in list_of_classes or "EMPTY" not in list_of_classes:
            self.list_of_classes.append("empty")
            kempty = len(self.list_of_classes)
            self.class_dict.update({kempty-1: "empty"})
        
    def get_list_of_classes(self):
        return self.list_of_classes, self.class_dict

    def get_list_of_propositions(self):
        return self.list_of_propositions, self.prop_dict
        
    def custom_propositions(self):
        # Class to set custom propositions.
        pass

    ### ------- End of Propositions ----------- ####

    ### ------- Code to compute class labeled confusion matrices ----------- ####

    def get_distance_param_conf_mat(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Get a dictionary with the distance parametrized confusion matrices for each distance bin.
        
        Args:
            None
        
        Returns:
            A dictionary where the keys are tuples of the form (lower_dist_thresh, upper_dist_thresh)
            The values are the corresponding distance parameterized confusion matrices. 
        """
        for key in list(self.disc_gt_boxes.keys()):
            self.dist_conf_mats[key] = self.compute_class_labeled_cm(self.disc_gt_boxes[key], self.disc_pred_boxes[key], self.conf_mat_mapping)

        return self.dist_conf_mats
    
    def compute_class_labeled_cm(self,
                            gt_boxes:EvalBoxes, 
                            pred_boxes: EvalBoxes, 
                            conf_mat_mapping: Dict,
                            dist_thresh: float = 2.0,       # in m 
                            yaw_thresh: float = np.pi/2.0): # in radians  -> np.ndarray:

        EMPTY = len(self.list_of_classes)
        distance_param_conf_mat = np.zeros( (len(self.list_of_classes)+1, len(self.list_of_classes)+1) )
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
                            
            # print(len(sample_pred_list))
            # if self.validation and (sample_token in list_of_validation_tokens):
            #         render_sample_data_with_predictions(self.nusc.get('sample', sample_token)['data']['LIDAR_TOP'], sample_pred_list, nusc=self.nusc)
                
        return distance_param_conf_mat 

    ### ------- End of Code to compute class labeled confusion matrices ----------- ####

    ### ------- Code to compute proposition labeled confusion matrices ----------- ####
    
    def get_prop_labeled_cm(self):
        """Get a dictionary with the proposition labelled confusion matrices for each distance bin.
        
        Args:
            None
            
        Returns:
            A dictionary where the keys are tuples of the form (lower_dist_thresh, upper_dist_thresh)
            The values are the corresponding proposition labelled confusion matrices.
        """
        n = len(self.list_of_propositions)

        # Looping over radius bands
        for key in list(self.disc_gt_boxes.keys()):
            self.prop_conf_mats[key] = np.zeros((n,n))

            # Loop over samples:
            for sample_token in self.sample_tokens:
                evaluation = self.single_evaluation_prop_cm(self.disc_gt_boxes[key][sample_token], 
                                                self.disc_pred_boxes[key][sample_token])
                self.prop_conf_mats[key] += evaluation
    
        return self.prop_conf_mats

    def get_clustered_conf_mat(self):   
        n = len(self.list_of_propositions)
        for radius_band in list(self.gt_clusters.keys()): # -> loop through distance params
            self.clustered_conf_mats[radius_band] = np.zeros((n,n))
            # Loop over samples:
            for sample_token in self.sample_tokens:
                # Radius band object. 
                radius_band_obj = self.gt_clusters[radius_band][sample_token]
                for cluster in radius_band_obj.clusters:
                    pred_boxes_in_cluster = self.find_preds_for_cluster(cluster, dist_thresh=2.0)
                    evaluation = self.single_evaluation_prop_cm(cluster.boxes, pred_boxes_in_cluster) # in radians
                    self.clustered_conf_mats[radius_band] += evaluation

    
    def find_preds_for_cluster(self, cluster:Cluster,
                               dist_thresh = None, 
                               yaw_thresh: float = np.pi/2.0,
                               iou_thresh:float = 0.60) -> EvalBoxes:
        '''
        Move this code to the cluster file.
        Inputs:
        TODO: Describe the inputs and outputs
        cluster: Cluster object.
        dist_thresh: float
        yaw_thresh: float
        iou_thresh: float
        '''
        
        dist_thresh = cluster.max_dist_bw_obj if (dist_thresh is None) else dist_thresh
        inrange_pred_boxes = self.disc_pred_boxes[cluster.radius_band][cluster.sample_token]
        sample = self.nusc.get('sample', cluster.sample_token)
        matched_pred_boxes_as_DetBoxes = EvalBoxes()
        matched_pred_box_idx = []
        matched_pred_boxes = []
        # st()
        for gt_idx, gt_box in enumerate(cluster.boxes):
            
            for pred_idx, pred in enumerate(inrange_pred_boxes):
                #TODO: currently this returns a Box object. I create the Box object in this function. Can we get away with creating a DetectionBox obj?
                ego_pred_box = convert_EvalBox_to_flat_veh_coords(sample_data_token=sample["data"]["LIDAR_TOP"], 
                                                                        box=pred, 
                                                                        nusc=self.nusc)
                ego_angle = np.arctan2(ego_pred_box.center[1], ego_pred_box.center[0])
                ego_angle = ego_angle if ego_angle >= 0 else (2 * np.pi) + ego_angle
                pred_idx = int(np.ceil(ego_angle / self.radius_band[0]))
                
                assert ego_pred_box.label in class_names, "Error: gt_box.detection_name not in list_of_classes"
                
                if cluster.lower_radian_lim <= ego_angle <= cluster.upper_radian_lim:
                    if center_distance(pred, gt_box) < dist_thresh and yaw_diff(pred, gt_box) < yaw_thresh and scale_iou(inrange_pred_boxes[match_idx], gt_box) > iou_thresh:
                        matched_pred_box_idx.append(pred_idx)
            
            # st()
            for match_idx in matched_pred_box_idx:
                matched_box:Box = inrange_pred_boxes[match_idx]
                #TODO how to avoid this object creation in line
                matched_detbox=[DetectionBox(sample_token=cluster.sample_token,
                                    translation=matched_box.center,
                                    size=matched_box.wlh,
                                    rotation=matched_box.orientation,
                                    detection_name=matched_box.name,
                                    detection_score=matched_box.score)]
                matched_pred_boxes_as_DetBoxes.add_boxes(sample_token=cluster.sample_token, boxes=matched_detbox)
                matched_pred_boxes.append(matched_detbox)
                # st()
        return matched_pred_boxes            
    
    def single_evaluation_prop_cm(self, gt_boxes:EvalBoxes, pred_boxes: EvalBoxes) -> np.ndarray:
        # single evaluation for proposition labeled confusion matrix
        n = len(self.list_of_propositions)
        prop_cm = np.zeros((n,n))   
        gt_idx, pred_idx = self.get_prop_cm_indices(gt_boxes, pred_boxes)
        if (gt_idx is not None) and (pred_idx is not None):
            prop_cm[pred_idx][gt_idx] += 1   
        if (gt_idx is None and pred_idx is not None) or (pred_idx is None and gt_idx is not None):
            print("Error: One of the confusion matrix indices is returned as None. Check")
        return prop_cm
    
    ### ---- Finding indices for proposition labeled confusion matrices ----- ###

    def get_prop_cm_indices(self, gt_boxes, pred_boxes):
        """
        Returns the predicted and ground truth indices of a confusion
        matrix for the given set of gt_classes and pred_classes
        gt_boxes: List[Box]
        pred_boxes: List[Box]
        """
        gt_idx = None
        pred_idx = None

        try:    
            gt_classes = {gt.detection_name for gt in gt_boxes}
            pred_classes = {pred.detection_name for pred in pred_boxes}
        except:
            st()

        # TODO: Use a conf_mat_mapping to make this more generic
        gt_classes = set({"ped" if x == "pedestrian" else "obs" for x in gt_classes})
        pred_classes = set({"ped" if x == "pedestrian" else "obs" for x in pred_classes})

        # # Conf_mat mapping:
        # # ToDo check if the following works correctly.
        # gt_classes = set({self.conf_mat_mapping[gt_box.detection_name] for gt_box in gt_boxes})
        # pred_classes = set({self.conf_mat_mapping[pred_box.detection_name] for pred_box in pred_boxes})
        

        if len(gt_classes) == 0:
            gt_classes = set({"empty"})
        if len(pred_classes) == 0:
            pred_classes = set({"empty"})

        for k, prop_label in self.prop_dict.items():
            if gt_classes == prop_label:
                gt_idx = k
            if pred_classes == prop_label:
                pred_idx = k
        return gt_idx, pred_idx

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