# Script to generate confusion matrix for class labels

# Copy of the evaluate.py script from nuScenes
import os
import numpy as np
from cluster import Cluster
from collections.abc import Iterable
from typing import Tuple, Dict, Any, List
from itertools import chain, combinations

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
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
        # self.lower_thresh = lower_thresh
        # self.upper_thresh = upper_thresh
        self.distance_bin = distance_bin
        self.max_dist = max_dist
        self.num_bins = int(max_dist // distance_bin)
        self.list_of_classes = list_of_classes
        self.verbose = verbose
        self.max_dist_bw_obj = max_dist_bw_obj
        self.dist_conf_mats: dict(Tuple[int, int], np.ndarray) = {}
        self.prop_conf_mats: dict(Tuple[int, int], np.ndarray) = {}
        self.disc_gt_boxes: dict(Tuple[int, int], EvalBoxes) = {}
        self.disc_pred_boxes: dict(Tuple[int, int], EvalBoxes) = {}
        self.conf_mat_mapping = conf_mat_mapping
        
        self.gt_clusters:dict(Tuple[int, int], dict(str, list)) = {}   # {distance_bin: {sample_token: [Cluster1, Cluster2, ...]}
        self.pred_clusters:dict(Tuple[int, int], dict(str, list)) = {}   # {sample_token: {distance_bin: [Cluster1, Cluster2, ...]}}
        
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
        
        n = len(self.list_of_classes)
        
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
        """
        
        '''Hierarchy
            distance_bin: (0, 10), (11, 20), (21, 30), ... (91, 100) if distance_bin = 10
                sample_token:
                    Cluster1, Cluster2 ... Calculated according to s = r*theta where s is self.max_dist_bw_obj and r is the smaller radius

        '''
        for i in range(self.num_bins):
            if i = 0:
                self.gt_clusters[(0, self.distance_bin)] = {}  
                cluster_theta = self.__calculate_max_radius_bw_obj()
                    
                for sample_token in self.gt_boxes.sample_tokens:
                    self.gt_clusters[(0, self.distance_bin)][sample_token] = []
                    
                    for j in range(np.ceil(np.pi/cluster_theta)):
                        self.gt_clusters[(0, self.distance_bin)][sample_token].append(Cluster(sample_token, 
                                                                                              None, 
                                                                                              self.max_dist_bw_obj, 
                                                                                              (0, self.distance_bin), ))
            else:
                self.gt_clusters[( (self.distance_bin*i)+1, (self.distance_bin*(i+1)))] = {}
        
        
        self.gt_clusters[(0, self.distance_bin)] = {}
        self.pred_clusters[(0, self.distance_bin)] = {}
                for sample_token in self.gt_boxes.sample_tokens:
                    self.gt_clusters[(0, self.distance_bin)][sample_token] = []
                    self.pred_clusters[(0, self.distance_bin)][sample_token] = []
        
                    
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
            self.prop_conf_mats[key] = self.calculate_prop_labelled_conf_mat(self.disc_gt_boxes[key], self.disc_pred_boxes[key], ["ped", "obs"], self.list_of_classes)
    
        return self.prop_conf_mats
    
    def get_clustered_conf_mat(self):
        self.cluster(self.disc_gt_boxes[(11, 20)], 
                     self.disc_pred_boxes[(11, 20)], 
                     (11, 20), 
                     self.list_of_classes)
    

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
    
    def cluster(self,
                gt_boxes:EvalBoxes, 
                pred_boxes: list, 
                dist_bin: Tuple[int, int],
                list_of_classes: list) -> np.ndarray:
        
        n = len(self.list_of_classes)
        # gt_vectors:Dict(EvalBox, np.ndarray) = {}
        # pred_vectors:Dict(EvalBox, np.ndarray) = {}
        
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
            ego_veh = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
            
            
            sample_pred_list = pred_boxes[sample_token]
            sample_gt_list = gt_boxes[sample_token]
            
            
            for gt in sample_gt_list:
                # Calculate the distance between the two by calculating the vector between the ego vehicle and the ground truth box
                # Go through the Average center of mass for each cluster. 
                #   If the average center of mass is within a threshold, add it to this, else, create a new cluster. Clusters are sorted counterclockwise
                #   Update the average center of mass for each cluster, AND check (first_elem - last elem) is still within threshold distance
                added = False
                
                for cluster in self.gt_clusters[dist_bin][sample_token]:
                    if cluster.can_add_box(gt):
                        cluster.add_box(gt)
                        added = True
                        break
                
                if added:
                    cluster = Cluster(sample_token, ego_veh, 2.0)
                    self.gt_clusters[dist_bin][sample_token].append(cluster)
                    
            
            for pred in sample_pred_list:
                added = False
                
                for cluster in self.pred_clusters[dist_bin][sample_token]:
                    if cluster.can_add_box(pred):
                        cluster.add_box(pred)
                        added = True
                        break
                
                if not added:
                    cluster = Cluster(sample_token, ego_veh, 2.0)
                    cluster.add_box(pred)
                    self.pred_clusters[dist_bin][sample_token].append(cluster)
        
         
        
    
    def calculate_prop_labelled_conf_mat(self, 
                                         gt_boxes:EvalBoxes, 
                                         pred_boxes: list, 
                                         list_of_propositions: list, 
                                         class_names:list) -> np.ndarray:
        
        n = len(self.list_of_classes)
        propn_labelled_conf_mat = np.zeros( ( (2**n), (2**n)) )

        propn_indices = list(range(len(list_of_propositions)))
        propn_powerset = list(self.powerset(propn_indices))

        for sample_token in gt_boxes.sample_tokens:
            sample_pred_list = pred_boxes[sample_token]
            sample_gt_list = gt_boxes[sample_token]
            taken = set()  # Initially no gt bounding box is matched.

            gt_classes = {gt.detection_name for gt in sample_gt_list}
            pred_classes = {pred.detection_name for pred in sample_gt_list}

            #TODO convert into generic
            gt_classes = {"ped" if x == "pedestrian" else "obs" for x in gt_classes}
            pred_classes = {"ped" if x == "pedestrian" else "obs" for x in pred_classes}

            gt_idx = 0
            pred_idx = 0

            for i, propn in enumerate(propn_powerset):

                classes = {} if len(propn) == 0 else {list_of_propositions[c] for c in propn}

                if gt_classes == set(classes):
                    gt_idx = i
                if pred_classes == set(classes):
                    pred_idx = i

            propn_labelled_conf_mat[pred_idx][gt_idx] += 1
            
        return propn_labelled_conf_mat
        
