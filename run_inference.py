import os
import subprocess
from pathlib import Path

pcd_path = Path("~/nuscenes_dataset/nuscenes_mini/sweeps/LIDAR_TOP/")
pcd_list = os.listdir(str(pcd_path))
# subprocess.run(cwd="")