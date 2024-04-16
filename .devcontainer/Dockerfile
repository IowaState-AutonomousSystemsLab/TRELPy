FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

WORKDIR /root
## Setting up dependencies
# ENV PATH="$PATH:/root/stormpy:/root/carl_storm:/root/storm:/root/pycarl:/root/tulip-control:/root/.local"
RUN apt-get update && \
      apt-get -y install sudo python3-pip python-is-python3 vim git

RUN mkdir /root/software
WORKDIR /root/software

ARG DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true

### Setting up all the necessary respositories 
RUN apt-get install -y --reinstall ca-certificates

RUN sudo apt-get update && apt-get install -qy software-properties-common && \
    sudo add-apt-repository ppa:deadsnakes/ppa

RUN sudo apt-get update && \
    sudo apt install -qy curl python3.8-dev python3.8-distutils && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    cd ~/ && \
    curl https://bootstrap.pypa.io/get-pip.py -o ~/get-pip.py && \
    python get-pip.py

RUN sudo apt-get -qy update
RUN sudo apt-get install -qy apt-transport-https
RUN sudo apt-get install -qy default-jre
RUN sudo apt-get install -qy wget \
					default-jdk \
					git \
					cmake \
					build-essential \
					libgmp-dev \
					libeigen3-dev \
					libboost-all-dev \
					ginac-tools \
					libginac-dev \
					libcln-dev \
					autoconf \
					glpk-utils \
					hwloc \
					libglpk-dev \
					libhwloc-dev \
					libz3-dev \
					libxerces-c-dev \
					automake \
					libz3-dev \ 
					graphviz \
                    software-properties-common

RUN sudo add-apt-repository ppa:ubuntu-toolchain-r/test

RUN sudo apt update
RUN sudo apt install build-essential
RUN apt install build-essential manpages-dev software-properties-common -qy && add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN sudo apt install gcc-13 g++-13 -y
RUN sudo apt install libmpfr-dev libgmp3-dev libmpc-dev -y
RUN sudo unlink /usr/bin/gcc && ln -s /usr/bin/gcc-13 /usr/bin/gcc
RUN sudo unlink /usr/bin/g++ && ln -s /usr/bin/g++-13 /usr/bin/g++

# # Cloning necessary repos i and installing things

RUN wget https://www.prismmodelchecker.org/dl/prism-4.8-linux64-x86.tar.gz
RUN tar -xvf prism-4.8-linux64-x86.tar.gz
WORKDIR /root/software/prism-4.8-linux64-x86
RUN bash install.sh
RUN echo "alias prism='bash /root/software/prism-4.8-linux64-x86/bin/prism'" >> ~/.bashrc
RUN echo "alias xprism='bash /root/software/prism-4.8-linux64-x86/bin/xprism'" >> ~/.bashrc


## Installing CARL
WORKDIR /root/software
RUN git clone https://github.com/moves-rwth/carl-storm
WORKDIR /root/software/carl-storm
RUN export STORM_DIR=/root/software/carl-storm && mkdir build
WORKDIR /root/software/carl-storm/build
RUN cmake .. && make lib_carl -j8


RUN pip3 install numpy \
				seaborn \
				networkx==2.0 \
				pytest \
				numpy \
				scipy \
				cvxopt \
				gr1py

WORKDIR /root/software
RUN git clone https://github.com/moves-rwth/stormpy.git --branch 1.8.0
WORKDIR /root/software/stormpy
# RUN python3 setup.py develop


## Installing TULIP

# WORKDIR /root/software
# RUN git clone https://github.com/tulip-control/tulip-control.git
# RUN cd /root/software/tulip-control && pip3 install -r requirements.txt
# WORKDIR /root/software/tulip-control
# RUN pip3 install .

# RUN apt-get update

