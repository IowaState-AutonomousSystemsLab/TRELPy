import os
import subprocess
from datetime import datetime
from nuscenes.nuscenes import NuScenes
from config import dataset_version, dataset_root

now = datetime.now()
time_at_run = now.strftime("%Y-%m-%d_%H-%M")

nusc = NuScenes(version=dataset_version, dataroot = dataset_root)

def getGitRoot():
    """Gets the root directory of the git repository

    Returns:
        str: path the denotes the root directory of the git repository
    """
    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} not found. Creating...")
        os.makedirs(dir_path)
    else:
        print(f"Not creating {dir_path} because it already exists")