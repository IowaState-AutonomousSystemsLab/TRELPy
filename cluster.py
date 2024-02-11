#! /usr/bin/python3

import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox

class Clusters:
    """Given a distance threshold and a radius band, this class consists of all the Cluster objects therein
    """
    def __init__(self,
                 sample_token:str,
                 ego_veh:dict,
                 radius_band:tuple,
                 max_distance_bw_obj:float,    
                ) -> None:
        """
        Args:
            sample_token (str): token of the sample this Cluster belongs to
            ego_veh (dict): ego vehicle dictionary. Contains translation(x,y,z), rotation(Quaternion), size(x,y,z), etc.
            radius_band (tuple): tuple of the radius band, (min_radius, max_radius) between which these Clusters lies
            max_distance_bw_obj (float): maximum distance between objects
        """
        
        self.sample_token = sample_token
        self.ego_veh = ego_veh
        self.radius_band = radius_band
        self.clusters: list(Cluster) = []
    
        self.generate_clusters()
        
        
    def generate_clusters(self):
        """generates clusters for the ground truth boxes
        """
        if self.radius_band[0] <= 0:
            raise ValueError("Radius band should be greater than 0")
        
        cluster_radial = self.__calculate_max_radius_bw_obj(self.radius_band[0])
        
        for i in range(np.ceil(self.radius_band[0]/(2 * np.pi))):
            self.clusters.append(Cluster(sample_token=self.sample_token,
                                         ego_veh=self.ego_veh,
                                         dist_threshold=self.dist_threshold,
                                         radius_band=cluster_radial))
        
        
    
    def __calculate_max_radius_bw_obj(self, radius: float):
        return (self.max_dist_bw_obj / radius)

class Cluster:
    """
    """
    def __init__(self, 
                 sample_token: str,
                 ego_veh,
                 dist_threshold, 
                 radius_band
                 lower_radian_lim, 
                 upper_radian_lim) -> None:
        """

        """
        self.sample_token = sample_token
        self.distance_threshold = dist_threshold
        self.center_of_mass = -1
        self.boxes: list(EvalBox) = []
        self.radius_band = None
        self.closest_box = None
        self.farthest_box = None
        self.ego_vehicle = ego_veh
        self.lower_radian_lim = lower_radian_lim
        self.upper_radian_lim = upper_radian_lim
        # self.ego_veh_yaw, _, _= Quaternion(ego_veh['rotation']).yaw_pitch_roll()   # Quaternion() returns a tuple not a class object?
        
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
    
        
    def add_box(self, box: EvalBox) -> None:
        self.boxes.append(box)
        
        
        # v = np.array(self.ego_vehicle['translation']) - np.array(box.translation)
        # angle = self.__angle_between(v, )
        
        # if len(self.boxes) == 0:
        #     self.closest_box = box
        #     self.farthest_box = box
        # else:
            
            
        
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
        
    
    