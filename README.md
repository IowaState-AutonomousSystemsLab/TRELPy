# TRELPy
Toolbox for **T**ask **R**elevant **E**va**L**uation of **P**erception in **Py**thon.

## Introduction

In safety-critical autonomous systems, deriving system-level guarantees requires evaluation of individual subsystems in a manner consistent with the system-level task. These safety guarantees require careful reasoning about how to evaluate each subsystem, and the evaluations have to be consistent with subsystem interactions and any assumptions made therein. A common example is the interaction between perception and planning. TRELPy is a Python-based toolbox that can evaluate the performance of perception models and leverage these evaluations in computing system-level guarantees via probabilistic model checking. The tool implements this framework for popular detection metrics such as confusion matrices, and implements new metrics such as proposition-labeled confusion matrices. The propositional formulae for the labels of the confusion matrix are chosen such that the confusion matrices are relevant to the downstream planner and system-level task. TRELPy can also group objects by egocentric distance or by orientation relative to the ego vehicle to further make the confusion matrix more task relevant. These metrics are leveraged to compute the combined performance of the perception and planner to get probabilistic system-level guarantees.

This toolbox contains utilities to load a pre-trained model, run inference on a custom dataset, generate confusion matrices, and compute probabilities for satisfaction of system-level tasks.
&nbsp; 
&nbsp;
<img src="figures/tool_flowchart.png" align="center"> 
&nbsp;

`tutorial.ipynb` walks through the a working example. The example uses the pretrained PointPillars LiDAR model from MMDetection3D for the perception model and the validation split of nuScenes as the dataset.

## Setup
The setup to run the complete pipeline is more complex (load dataset -> inference -> TRELPy) than the one for running `tutorial.ipynb` (download inference results -> TRELPy).

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

<details>
<summary><font size="+1"><b>Setup for full TRELPy functionality</b></font></summary>

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

</details>

## Cite this
In the rightmost column on the main GitHub Repository, you have the option to export the citation to this tool in `BibTex` format. For other formats, see [this great GitHub repository](https://github.com/citation-file-format/cffconvert)

 