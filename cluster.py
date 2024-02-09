#! /usr/bin/python3

import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox

class Cluster:
    def __init__(self, 
                 sample_token: str,
                 ego_veh,
                 dist_threshold) -> None:
        self.center_of_mass = -1
        self.sample_token = sample_token
        self.boxes: list(EvalBox) = []
        self.dist_thresh = dist_threshold
        self.ego_vehicle = ego_veh
        self.ego_veh_yaw, _, _= Quaternion(ego_veh['rotation']).yaw_pitch_roll() 
        self.closest_box = None
        self.farthest_box = None
        
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
        
        return (x, y, z)
    
    def can_add_box(self, coord:np.ndarray):
        com = self.calculate_center_of_mass()
        dist = np.linalg.norm(com - coord)
        
        return dist < self.dist_thresh
        
    
    