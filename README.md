# TRELPy
Toolbox for **T**ask **R**elevant **E**va**L**uation of **P**erception in **Py**thon.

This toolbox contains utilities to load a pre-trained model, run inference, generate confusion matrices, and compute probabilities for satisfaction of system level. The user can specify specifications, and examples. 

`tutorial.ipynb` walks through the entire pipeline. It is trained on the PointPillars LiDAR model from MMDetection3D on the validation split of nuScenes. 
# Setup

All apt requirements are listed in `requirements-apt.txt` and can be downloaded using `sed 's/#.*//' apt-requirements.txt | xargs sudo apt-get -y install` \
All pip requirements are listed in `requirements-pip.txt` and can be downloaded using `pip3 install -r pip-requirements.txt`

 