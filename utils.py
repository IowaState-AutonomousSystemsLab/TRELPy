# Adapted from nuScenes devkit
# File created to add IDs to eval boxes so predictions can be queried easily

import abc
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from nuscenes.eval.detection.data_classes import DetectionBox

class EvalBox_ID(DetectionBox):
    def __init__(self, eval_box_instance, ID):
        super().__init__(**eval_box_instance.__dict__)
        self.ID = ID
