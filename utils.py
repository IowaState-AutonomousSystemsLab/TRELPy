# Adapted from nuScenes devkit
# File created to add IDs to eval boxes so predictions can be queried easily

import abc
from collections import defaultdict
from typing import List, Tuple, Union
import base64

import numpy as np
from nuscenes.eval.detection.data_classes import DetectionBox

class EvalBox_ID(DetectionBox):
    def __init__(self, eval_box_instance, ID):
        super().__init__(**eval_box_instance.__dict__)
        self.ID = ID
        
    def __hash__(self) -> int:
        # TODO How to we create a unique ID hash for the box? 
        # base64.b64encode("(2.5 3 8) jasih3jfuq1".encode('utf-8')).decode('utf-8')
        pass
