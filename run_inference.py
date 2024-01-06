# LIDAR Inference:
import os
import subprocess
from pathlib import Path
import pdb

home_dir = str(Path.home())

setsize = "mini"

if setsize == "mini":
    dataset_name = "nuscenes-mini"
    out_dir = f"{home_dir}/nuscenes_dataset/inference_results_mini"
else:
    dataset_name = "nuscenes"
    out_dir = f"{home_dir}/nuscenes_dataset/inference_results"
    
pcd_path = f"{home_dir}/software/mmdetection3d/data/{dataset_name}/samples/LIDAR_TOP/"

pcd_list = os.listdir(pcd_path)
print(len(pcd_list))

# configs_path = "configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
# checkpoint_path = "checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"

configs_path = "configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-amp-2x_nus-3d.py"
checkpoint_path = "checkpoints/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth"

for i, pcd in enumerate(pcd_list):
    path = Path(f"{pcd_path}/{pcd}").absolute()
    # print(path)
    if path.exists():
        cmd = f'python3 demo/pcd_demo.py {str(path)} {configs_path} {checkpoint_path} --device cuda --out-dir {out_dir}'
        
    subprocess.run(cmd, cwd=f"{home_dir}/software/mmdetection3d/", shell=True)
    
    if i%100 == 0:
        print(f"---- ---- !-!-!-!- run_inference.py: Done with {i} files")
