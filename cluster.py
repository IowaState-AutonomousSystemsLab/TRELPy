#! /usr/bin/python3

import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox

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
    

class RadiusBand:
    """Given a distance threshold between objects and a radius band, this class consists of all the Cluster objects therein
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
        self.max_dist_bw_obj = max_distance_bw_obj
        self.generate_clusters()
        
        self.sigma = 0.0001
        
        
    def generate_clusters(self):
        """generates clusters for the ground truth boxes
        """
        if self.radius_band[0] <= 0:
            raise ValueError("Radius band should be greater than 0")
        
        cluster_radial = self.__calculate_max_radius_bw_obj(self.radius_band[0])
        
        for i in range(int(np.ceil(self.radius_band[0]/(2 * np.pi)))):
            self.clusters.append(Cluster(sample_token=self.sample_token,
                                         ego_veh=self.ego_veh,
                                         dist_threshold=self.max_dist_bw_obj,
                                         radius_band=self.radius_band,
                                         lower_radian_lim=(0 if i==0 else (i*cluster_radial)+self.sigma),
                                         upper_radian_lim=(i+1)*cluster_radial)
            )
    
    def add_box(self, box: EvalBox) -> None:
        # Convert Quaternion into yaw, pitch, roll
        ego_veh_quat = Quaternion(self.ego_veh['rotation'])
        ego_veh_yaw, _, _ = ego_veh_quat.yaw_pitch_roll
        
        #Calculate the angle between the box and the ego vehicle
        obj_yaw = self.ego_veh[translation] - box.translation
        
    
    def __calculate_max_radius_bw_obj(self, radius: float):
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
        self.boxes: list(EvalBox) = []
        self.radius_band = None
        self.ego_vehicle = ego_veh
        self.lower_radian_lim = lower_radian_lim
        self.upper_radian_lim = upper_radian_lim

        # self.farthest_box = None
        # self.closest_box = None
        # self.center_of_mass = -1
        # self.ego_veh_yaw, _, _= Quaternion(ego_veh['rotation']).yaw_pitch_roll()   # Quaternion() returns a tuple not a class object?
        
    
        
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
        
    
    