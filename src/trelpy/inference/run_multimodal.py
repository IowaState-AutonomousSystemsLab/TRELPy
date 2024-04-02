import subprocess
from pathlib import Path

home_dir = str(Path.home())
dataset_root = "/home/apurvabadithela/software/mmdetection3d/data/nuscenes"
demo_file = "projects/BEVFusion/demo/multi_modality_demo.py"
pkl_file = "nuscenes_infos_val.pkl"
cam_type = "CAM_FRONT"
img_folder=f"{dataset_root}/samples/{cam_type}"
img_file=f"{img_folder}/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984233512470.jpg"
pcd_folder=f"{dataset_root}/samples/LIDAR_TOP"
pkl_file_loc = f"{dataset_root}/{pkl_file}"
checkpoint = "checkpoints/bevfusion_converted.pth"
config = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
out_dir="/home/apurvabadithela/software/mmdetection3d/outputs"

"""
samples/RADAR_FRONT/n015-2018-10-08-15-36-50+0800__RADAR_FRONT__1538984233560834.pcd
samples/RADAR_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__RADAR_FRONT_LEFT__1538984233559954.pcd
samples/RADAR_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__RADAR_FRONT_RIGHT__1538984233580978.pcd
samples/RADAR_BACK_LEFT/n015-2018-10-08-15-36-50+0800__RADAR_BACK_LEFT__1538984233568518.pcd
samples/RADAR_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__RADAR_BACK_RIGHT__1538984233550067.pcd
samples/LIDAR_TOP/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984233547259.pcd.bin
samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984233512470.jpg
samples/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984233520339.jpg
samples/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984233527893.jpg
samples/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984233537525.jpg
samples/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984233504844.jpg
samples/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984233547423.jpg
"""

command = f"python {demo_file} {pcd_folder} {img_folder} {pkl_file_loc} {config} {checkpoint} --out-dir {out_dir}"
print(command)

subprocess.run(command, cwd=f"{home_dir}/software/mmdetection3d/", shell=True)