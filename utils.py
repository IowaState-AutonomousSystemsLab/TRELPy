
#!/usr/bin/env python3

from typing import Iterable
from itertools import chain, combinations
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox

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
    
    print(f"Rotation of an EvalBox {(eval_box.rotation)}")

    box = Box(
        center=eval_box.translation,
        size=eval_box.size,
        orientation=eval_box.rotation,
        velocity=eval_box.velocity,
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
        translation = box.center,
        size = box.wlh,
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