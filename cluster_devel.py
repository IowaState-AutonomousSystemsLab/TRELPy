# Taken from clustering branch (2/25/24)

#! /usr/bin/python3

import sys
import numpy as np
# import pytestp
from typing import List
from pyquaternion import Quaternion
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.utils.data_classes import Box
from pdb import set_trace as st

class RadiusBand:
    """Given a distance threshold between objects and a radius band, this class consists of all the Cluster objects therein
    """
    def __init__(self,
                 sample_token:str,
                 gt_boxes, 
                 ego_veh:dict,
                 radius_band:tuple,
                 max_distance_bw_obj:float, 
                 verbosity:bool=True) -> None:   
        """
        Args:
            sample_token (str): token of the sample this Cluster belongs to
            ego_veh (dict): ego vehicle dictionary. Contains translation(x,y,z), rotation(Quaternion), size(x,y,z), etc.
            radius_band (tuple): tuple of the radius band, (min_radius, max_radius) between which these Clusters lies
            max_distance_bw_obj (float): maximum distance between objects
        """
        # print("Creating RadiusBand object for sample: ", sample_token, " with radius band: ", radius_band)
        self.sample_token = sample_token
        self.ego_veh = ego_veh
        self.radius_band = radius_band
        self.clusters: list[Cluster] = []
        self.max_dist_bw_obj = max_distance_bw_obj

        
        if radius_band[0] <= 0:
            print("Minimum radius should be greater than 0\n setting min_radius to 1", file=sys.stderr)
            self.radius_band = (1, radius_band[1])
        
        self.sigma = 0.0001 #TODO Remove sigma and ensure all ranges are accounted for
        self.angular_diff = self.__calculate_max_radius_bw_obj(self.radius_band[0])
        self.num_clusters = int(np.ceil((2 * np.pi) / self.angular_diff))
        self.angular_diff = (2 * np.pi / self.num_clusters)
        
        self.generate_clusters()
        self.populate_clusters(gt_boxes)
        self.compute_total_objects_in_band()
    
    def compute_total_objects_in_band(self):
        total_gt_boxes_in_band = 0
        for cluster in self.clusters:
            total_gt_boxes_in_band += len(cluster.boxes)
        # print("Total number of objects in band: ", str(total_gt_boxes_in_band))
        # if total_gt_boxes_in_band > 0:
            # st()

    def generate_clusters(self):
        """generates clusters for the ground truth boxes
        """
        
        for i in range(self.num_clusters):
            # if i%100 == 0: print(f"Creating cluster {i+1} of {self.num_clusters}")
            self.clusters.append(Cluster(sample_token=self.sample_token,
                                         ego_veh=self.ego_veh,
                                         dist_threshold=self.max_dist_bw_obj,
                                         radius_band=self.radius_band,
                                         lower_radian_lim=(0 if i==0 else (i*self.angular_diff)+self.sigma),
                                         upper_radian_lim=(i+1)*self.angular_diff)
            )
    
    def populate_clusters(self, unclustered_gt_boxes: list) -> None:
        """populates the clusters with the ground truth boxes
        """
        for box in unclustered_gt_boxes:
            self.add_box(box)
        
    
    def add_box(self, box: Box) -> None:
        angle_from_ego = np.arctan2(box.center[1], box.center[0])
        angle_from_ego = angle_from_ego if angle_from_ego >= 0 else (np.pi * 2.0) + angle_from_ego
        
        bin_index = int(np.floor(angle_from_ego / self.angular_diff))
        try:
            assert bin_index < len(self.clusters) and bin_index >= 0
            self.clusters[bin_index].add_box(box)
        except:
            st()
        
        
    
    def __calculate_max_radius_bw_obj(self, radius: float):
        """ Using s=r * theta, calculate the maximum distance between objects
        
        Since we know r (the radius from the ego_veh), and the max allowed dist distance between objects, we calculate theta
        """
        #TODO: I'm using arc length formula for straight line distance. 
        return (self.max_dist_bw_obj / radius)

class Cluster:
    """
    """
    def __init__(self, 
                 sample_token: str,
                 ego_veh,
                 dist_threshold, 
                 radius_band,
                 lower_radian_lim, 
                 upper_radian_lim) -> None:
        """

        """
        self.sample_token = sample_token
        self.distance_threshold = dist_threshold
        self.boxes: List = []
        self.radius_band = radius_band
        self.ego_vehicle = ego_veh
        self.lower_radian_lim = lower_radian_lim
        self.upper_radian_lim = upper_radian_lim
        # self.farthest_box = None
        # self.closest_box = None
        # self.center_of_mass = -1
        # self.ego_veh_yaw, _, _= Quaternion(ego_veh['rotation']).yaw_pitch_roll()   # Quaternion() returns a tuple not a class object?
        
    
        
    def add_box(self, box: Box) -> None:       
        angle_from_ego = np.arctan2(box.center[1], box.center[0])
        angle_from_ego = angle_from_ego if angle_from_ego >= 0 else (np.pi * 2.0) + angle_from_ego
        
        if self.lower_radian_lim <= angle_from_ego <= self.upper_radian_lim:
            self.boxes.append(box)
        else:
            print("Box not added!")

    
    def get_num_items_in_cluster(self) -> int:
        return len(self.boxes)
            
        
    def get_cluster_spread(self) -> np.ndarray:
        pass
        
    def calculate_center_of_mass(self) -> None:
        x = 0
        y = 0
        z = 0
        for box in self.boxes:
            x += box.translation[0]
            y += box.translation[1]
            z += box.translation[2]
        x /= len(self.boxes)
        y /= len(self.boxes)
        z /= len(self.boxes)
        
        self.center_of_mass = (x, y, z)
        
        return (x, y, z)
    
    def can_add_box(self, coord:np.ndarray):
        com = self.calculate_center_of_mass()
        dist = np.linalg.norm(com - coord)
        
        return dist < self.dist_thresh
    

def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
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
        
    
    