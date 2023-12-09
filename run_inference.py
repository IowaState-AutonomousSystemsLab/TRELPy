import os
import subprocess
from pathlib import Path
import pdb

home_dir = str(Path.home())
out_dir = f"{home_dir}/software/run_nuscenes_evaluations/outputs"

if os.path.isdir(out_dir):
    print("Exists")
else:
    print("Doesn't exists")
    os.mkdir(out_dir)

pcd_path = f"{home_dir}/software/mmdetection3d/data/nuscenes/samples/LIDAR_TOP/"
pcd_list = os.listdir(pcd_path)

configs_path = "configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
checkpoint_path = "checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"

for pcd in pcd_list[:10]:
    path = Path(f"{pcd_path}/{pcd}").absolute()
    # print(path)
    if path.exists():
        cmd = f'python3 demo/pcd_demo.py {str(path)} {configs_path} {checkpoint_path} --device cuda --out-dir {out_dir}'
    else:
        pdb.set_trace()
        
    subprocess.run(cmd, cwd=f"{home_dir}/software/mmdetection3d/", shell=True)