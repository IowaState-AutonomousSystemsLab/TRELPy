from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances,\
    get_panoptic_instances_stats
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap
import os.path as osp

from PIL import Image
from nuscenes.nuscenes import NuScenesExplorer 
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple, Union
from nuscenes.nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
import os 
from pdb import set_trace as st

## ==================== Debugging functions; not affecting main nuscenes_render ==================================== #
def render(box:Box, title:str,axis: Axes,view: np.ndarray = np.eye(3),normalize: bool = False, colors: Tuple = ('b', 'r', 'k'),
    linewidth: float = 2) -> None:
    """
    Renders the box in the provided Matplotlib axis.
    :param axis: Axis onto which the box should be drawn.
    :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
        back and sides.
    :param linewidth: Width in pixel of the box sides.
    """
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                    [corners.T[i][1], corners.T[i + 4][1]],
                    color=colors[2], linewidth=linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0])
    draw_rect(corners.T[4:], colors[1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    axis.plot([center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                color=colors[0], linewidth=linewidth)

    if "gt" in title:
        axis.text(center_bottom_forward[0]+1, center_bottom_forward[1]-1, title, fontsize='small')
    
    if "pred" in title:
        axis.text(center_bottom_forward[0]+1, center_bottom_forward[1]+1, title, fontsize='small')
        
def generate_fig_fn(out_path, sample_token):    
    folder_fn = os.path.join(out_path, "sample_token_" + sample_token)
    if not os.path.exists(folder_fn):
        os.makedirs(folder_fn)
    fig_fn = os.path.join(folder_fn, "main_mismatch_")
    i=1
    while os.path.isfile('{}{:d}.png'.format(fig_fn, i)):
        i += 1
    fig_fn = '{}{:d}.png'.format(fig_fn, i)
    return fig_fn

def convert_ego_pose_to_flat_veh_coords(sample_data_token:str, box:Box, nusc: NuScenes) -> Box:
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
    return box

def render_specific_gt_and_predictions(sample_token: str, gt_info:list=[], pred_info:list=[], nusc=None, out_path:str=None):
    '''
    Inputs:
    sample_data_token: token
    gt_info: List of tuples containing EvalBox and the associated label
    pred_info: List of tuples containing the EvalBox and the associated label
    nusc: nuscenes object
    out_path: plotting folder where the plot is saved

    # Default parameters:
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    '''
    with_anns: bool = True
    box_vis_level: BoxVisibility = BoxVisibility.ANY
    axes_limit: float = 40
    ax: Axes = None
    nsweeps: int = 1
    underlay_map: bool = True
    use_flat_vehicle_coordinates: bool = True
    show_lidarseg_legend: bool = False
    filter_lidarseg_labels: List = None
    lidarseg_preds_bin_path: str = None
    verbose: bool = True
    show_panoptic: bool = False

    fig_fn = generate_fig_fn(out_path, sample_token)

    nusc_exp = NuScenesExplorer(nusc)
    color_map = get_colormap()
    # Get sensor modality.
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample["data"]["LIDAR_TOP"]
    sd_record = nusc.get('sample_data', sample_data_token)
    sensor_modality = sd_record['sensor_modality']
    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    ref_chan = 'LIDAR_TOP'
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_record = nusc.get('sample_data', ref_sd_token)
    
    pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan,
                                                    nsweeps=nsweeps)
    velocities = None
    if use_flat_vehicle_coordinates:
        # Retrieve transformation matrices for reference point cloud.
        cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
        ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                        rotation=Quaternion(cs_record["rotation"]))

        # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
        ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        rotation_vehicle_flat_from_vehicle = np.dot(
            Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
            Quaternion(pose_record['rotation']).inverse.rotation_matrix)
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
        viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
    else:
        viewpoint = np.eye(4)

    # Init axes.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Render map if requested.
    if underlay_map:
        assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                'otherwise the location does not correspond to the map!'
        nusc_exp.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

    points = view_points(pc.points[:3, :], viewpoint, normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
    
    ax.plot(0, 0, 'x', color='red')
    
    # Show boxes.
    if with_anns:
        for (box, label) in gt_info:
            box = convert_ego_pose_to_flat_veh_coords(ref_sd_token, box, nusc)
            c = np.array(color_map[box.name]) / 255.0
            render(box, label, ax, view=np.eye(4), colors=(c, c, c))
            
        for (box, label) in pred_info:
            box = convert_ego_pose_to_flat_veh_coords(ref_sd_token, box, nusc)
            render(box, label, ax, view=np.eye(4), colors=('k', 'k', 'k'))

    ax.axis('off')
    ax.set_title('{} {labels_type}'.format(
        sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    ax.set_aspect('equal')

    if out_path is not None:
        plt.savefig(fig_fn, bbox_inches='tight', pad_inches=0, dpi=200)

    if verbose:
        plt.show()

def get_colormap():
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "pedestrian": (0, 0, 230),  # Blue
        "barrier": (112, 128, 144),  # Slategrey
        "traffic_cone": (222, 184, 135),  # Burlywood
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child": (135, 206, 235),  # Skyblue,
        "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
        "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
        "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
        "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (210, 105, 30),  # Chocolate
        "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
        "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy": (255, 127, 80),  # Coral
        "bus": (255, 127, 80),  # Coral
        "vehicle.bus.rigid": (255, 69, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "car": (255, 158, 0),  # Orange
        "vehicle.construction": (233, 150, 70),  # Darksalmon
        "vehicle.emergency.ambulance": (255, 83, 0),
        "vehicle.emergency.police": (255, 215, 0),  # Gold
        "vehicle.motorcycle": (255, 61, 99),  # Red
        "motorcycle": (255, 61, 99),  # Red
        "vehicle.trailer": (255, 140, 0),  # Darkorange
        "trailer": (255, 140, 0),  # Darkorange
        "vehicle.truck": (255, 99, 71),  # Tomato
        "truck": (255, 99, 71),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other": (175, 0, 75),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (112, 180, 60),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (255, 228, 196),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }

    return classname_to_color
## ==================== End of Debugging functions; not affecting main nuscenes_render ==================================== #

def convert_to_flat_veh_coords(sample_data_token:str, box:Box, nusc: NuScenes) -> Box:
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
    
    return box

def render_sample_data_with_predictions( sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 40,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None,
                           verbose: bool = True,
                           show_panoptic: bool = False,
                           pred_boxes: list=[],
                           nusc=None) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        
        nusc_exp = NuScenesExplorer(nusc)
        
        if show_lidarseg:
            show_panoptic = False
        # Get sensor modality.
        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                if show_lidarseg or show_panoptic:
                    gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                    assert hasattr(nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                    # Ensure that lidar pointcloud is from a keyframe.
                    assert sd_record['is_key_frame'], \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                    assert nsweeps == 1, \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                        'be set to 1.'

                    # Load a single lidar point cloud.
                    pcl_path = osp.join(nusc.dataroot, ref_sd_record['filename'])
                    pc = LidarPointCloud.from_file(pcl_path)
                else:
                    # Get aggregated lidar point cloud in lidar frame.
                    pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan,
                                                                     nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                nusc_exp.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            if sensor_modality == 'lidar' and (show_lidarseg or show_panoptic):
                gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                semantic_table = getattr(nusc, gt_from)
                # Load labels for pointcloud.
                if lidarseg_preds_bin_path:
                    sample_token = nusc.get('sample_data', sample_data_token)['sample_token']
                    lidarseg_labels_filename = lidarseg_preds_bin_path
                    assert os.path.exists(lidarseg_labels_filename), \
                        'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                        'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
                else:
                    if len(semantic_table) > 0:
                        # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                        lidarseg_labels_filename = osp.join(nusc.dataroot,
                                                            nusc.get(gt_from, sample_data_token)['filename'])
                    else:
                        lidarseg_labels_filename = None

                if lidarseg_labels_filename:
                    # Paint each label in the pointcloud with a RGBA value.
                    if show_lidarseg or show_panoptic:
                        if show_lidarseg:
                            colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                        nusc.lidarseg_name2idx_mapping, nusc.colormap)
                        else:
                            colors = paint_panop_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                            nusc.lidarseg_name2idx_mapping, nusc.colormap)

                        if show_lidarseg_legend:

                            # If user does not specify a filter, then set the filter to contain the classes present in
                            # the pointcloud after it has been projected onto the image; this will allow displaying the
                            # legend only for classes which are present in the image (instead of all the classes).
                            if filter_lidarseg_labels is None:
                                if show_lidarseg:
                                    # Since the labels are stored as class indices, we get the RGB colors from the
                                    # colormap in an array where the position of the RGB color corresponds to the index
                                    # of the class it represents.
                                    color_legend = colormap_to_colors(nusc.colormap,
                                                                    nusc.lidarseg_name2idx_mapping)
                                    filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)
                                else:
                                    # Only show legends for stuff categories for panoptic.
                                    filter_lidarseg_labels = stuff_cat_ids(len(nusc.lidarseg_name2idx_mapping))

                            if filter_lidarseg_labels and show_panoptic:
                                # Only show legends for filtered stuff categories for panoptic.
                                stuff_labels = set(stuff_cat_ids(len(nusc.lidarseg_name2idx_mapping)))
                                filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

                            create_lidarseg_legend(filter_lidarseg_labels,
                                                   nusc.lidarseg_idx2name_mapping,
                                                   nusc.colormap,
                                                   loc='upper left',
                                                   ncol=1,
                                                   bbox_to_anchor=(1.05, 1.0))
                else:
                    print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                          'from the ego vehicle instead.'.format(nusc.version))

            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            # scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
            # scatter = ax.scatter(0, 0, c=colors, s=point_scale)

            # Show velocities.
            # if sensor_modality == 'radar':
            #     points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
            #     deltas_vel = points_vel - points
            #     deltas_vel = 6 * deltas_vel  # Arbitrary scaling
            #     max_delta = 20
            #     deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            #     colors_rgba = scatter.to_rgba(colors)
            #     for i in range(points.shape[1]):
            #         ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(nusc_exp.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))
                    
                for box in pred_boxes:
                    box = convert_to_flat_veh_coords(ref_sd_token, box, nusc)
                    box.render(ax, view=np.eye(4), colors=('k', 'k', 'k'))

            # Limit visible range.
            # ax.set_xlim(-axes_limit, axes_limit)
            # ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(nusc_exp.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        ax.set_title('{} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()