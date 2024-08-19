# LIDAR Inference:
import os
import os.path as osp
import subprocess
from pathlib import Path
import pdb
from datetime import datetime
now = datetime.now()
from custom_env import *

# python projects/BEVFusion/demo/multi_modality_demo.py 
# demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin 
# demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl 
#  checkpoints/bevfusion_converted.pth --cam-type all --score-thr 0.2 --out-dir ~/nuscenes_dataset/test/test.png

home_dir = str(Path.home())
configs_path = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
checkpoint_path = "checkpoints/bevfusion_converted.pth"
setsize="full"
folder_name = "model_"+now.strftime("%m-%d-%Y_%H_%M")

if setsize == "mini":
    dataset_name = "nuscenes"
    out_dir = f"{home_dir}/nuscenes_dataset/inference_results_mini/"+folder_name
else:
    dataset_name = "nuscenes"
    out_dir = f"{home_dir}/nuscenes_dataset/inference_results/"+folder_name

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

info_file = os.path.join(out_dir, "model_info.txt")
with open(info_file, 'w') as f:
    f.write(f"configs_path = {configs_path} \n checkpoint_path = {checkpoint_path} \n")
    f.write(f"Detection threshold {DET_THRESH}")

    
pcd_path = f"{home_dir}/software/mmdetection3d/data/{dataset_name}/samples/LIDAR_TOP/"

pcd_list = os.listdir(pcd_path)
print(len(pcd_list))

def create_sample_dir(lidar_path):
    pass

if __name__ == "__main__":
    for i, pcd in enumerate(pcd_list):
        lidar_path = Path(f"{pcd_path}/{pcd}").absolute()
        # Create sample directory with all annotations
        sample_dir = create_sample_dir(lidar_path)
        print(lidar_path)
        pdb.set_trace()
        # if lidar_path.exists():
        #     if DET_THRESH == 0.3:
        #         cmd = f'python3 demo/pcd_demo.py {str(lidar_path)} {configs_path} {checkpoint_path} --device cuda --out-dir {out_dir}'
        #     else:
        #         cmd = f'python3 demo/pcd_demo.py {str(lidar_path)} {configs_path} {checkpoint_path} --device cuda --out-dir {out_dir} --pred-score-thr {DET_THRESH}'
            
        subprocess.run(cmd, cwd=f"{home_dir}/software/mmdetection3d/", shell=True)
        
        if i%10000 == 0:
            print(f"---- ---- !-!-!-!- run_inference.py: Done with {i} files")

    with open(info_file, 'a') as f:
        f.write(f"Inferences complete.")
        f.write(f"Detection threshold {DET_THRESH}")
    f.close()

    print(f" !-!-!-!- run_inference.py COMPLETE")