# LIDAR Inference:
import os
import subprocess
from pathlib import Path
import pdb
from datetime import datetime
now = datetime.now()

home_dir = str(Path.home())
setsize = "mini"
configs_path = "configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
checkpoint_path = "checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"

folder_name = "model_"+now.strftime("%m-%d-%Y_%H_%M")

if setsize == "mini":
    dataset_name = "nuscenes-mini"
    out_dir = f"{home_dir}/nuscenes_dataset/inference_results_mini/"+folder_name
else:
    dataset_name = "nuscenes"
    out_dir = f"{home_dir}/nuscenes_dataset/inference_results/"+folder_name

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

info_file = os.path.join(out_dir, "model_info.txt")
with open(info_file, 'w') as f:
    f.write(f"configs_path = {configs_path} \n checkpoint_path = {checkpoint_path} \n")
f.close()
    
pcd_path = f"{home_dir}/software/mmdetection3d/data/{dataset_name}/samples/LIDAR_TOP/"

pcd_list = os.listdir(pcd_path)
print(len(pcd_list))

for i, pcd in enumerate(pcd_list):
    path = Path(f"{pcd_path}/{pcd}").absolute()
    # print(path)
    if path.exists():
        cmd = f'python3 demo/pcd_demo.py {str(path)} {configs_path} {checkpoint_path} --device cuda --out-dir {out_dir}'
        
    subprocess.run(cmd, cwd=f"{home_dir}/software/mmdetection3d/", shell=True)
    
    if i%100 == 0:
        print(f"---- ---- !-!-!-!- run_inference.py: Done with {i} files")

with open(info_file, 'a') as f:
    f.write(f"Inferences complete.")
f.close()