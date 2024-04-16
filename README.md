# TRELPy
Toolbox for **T**ask **R**elevant **E**va**L**uation of **P**erception in **Py**thon.

## Introduction

This toolbox contains utilities to load a pre-trained model, run inference, generate confusion matrices, and compute probabilities for satisfaction of system level. The user can specify specifications, and examples. 

`tutorial.ipynb` walks through the entire pipeline. The examples use the pretrained PointPillars LiDAR model from MMDetection3D for the perception model and the validation split of nuScenes as the dataset.


## Setup
The setup to run the complete pipeline is more complex (dataset -> inference -> TRELPy -> Probabilistic Model Checking) than the one for running `tutorial.ipynb` (download inference results -> TRELPy).

We are in the process of developing a Docker container for these dependencies. Check back here to see updates.

### System Requirements for TRELPy
- OS: Ubuntu (Has been tested to work on Ubuntu 20.04)
- Computer with GPU to run inference

### Setup for `tutorial.ipynb`
The dependencies required to run the ipynb can be installed through the following commands.  

1. All apt requirements are listed in `requirements-apt.txt` and can be downloaded using 
    ```bash
    sed 's/#.*//' apt-requirements.txt | xargs sudo apt-get -y install
    ```


2. All pip requirements are listed in `requirements-pip.txt` and can be downloaded using 
    ```bash 
    pip3 install -r pip-requirements.txt
    ```

3. Ensure these programs are installed 
    1. [STORM](https://www.stormchecker.org/documentation/obtain-storm/build.html) 
    2. [STORMPy](https://moves-rwth.github.io/stormpy/installation.html) 
    3. [TuLiP](https://github.com/tulip-control/tulip-control)

### Setup for the full pipeline
1. Ensure `tutorial.ipynb` setup is completed
2. Install the required dependencies from the following table

|Name and link of program|What kind of installation| Versions tested on | Purpose |
|-|-|-|-|
| [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) | Local, Docker | cuda_12.4.r12.4 | Inference |
| [STORM](https://www.stormchecker.org/documentation/obtain-storm/build.html) | Local | - | TRELPy |
| [PyTorch](https://pytorch.org/get-started/locally/) | Local | 2.1.0+cu12 | Inference |
| [StormPy](https://moves-rwth.github.io/stormpy/installation.html) | Local |  | TRELPy |
| [TuLiP](https://github.com/tulip-control/tulip-control) |  Local |  | TRELPy |
| [MMDetection3D](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) | Local |  | Inference |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | Docker | 1.14.6 | Inference in Docker |
| [PRISM (Optional)](https://www.prismmodelchecker.org/manual/InstallingPRISM/Instructions) | Local |  | TRELPy |

**Local** means you are running on your ubuntu installation \
**Docker** means you will be using the provided Dockerfile. *This is currently a work in progress and not completely setup.*

## Cite this
In the rightmost column on the main GitHub Repository, you have the option to export the citation to this tool in `BibTex` format. For other formats, see [this great GitHub repository](https://github.com/citation-file-format/cffconvert)

 