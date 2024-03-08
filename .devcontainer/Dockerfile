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

RUN sudo apt install build-essential
RUN sudo apt install libmpfr-dev libgmp3-dev libmpc-dev -y
RUN wget http://ftp.gnu.org/gnu/gcc/gcc-13.2.0/gcc-13.2.0.tar.gz
RUN tar -xf gcc-13.2.0.tar.gz
RUN cd gcc-13.2.0
RUN bash /root/gcc-13.2.0/configure -v --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu --prefix=/usr/local/gcc-13.2.0 --enable-checking=release --enable-languages=c,c++ --disable-multilib --program-suffix=-13.2.0
RUN make -j3
RUN sudo make install
RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-13.2.0 13

# Cloning necessary repos i and installing things

RUN wget https://www.prismmodelchecker.org/dl/prism-4.8-linux64-x86.tar.gz
RUN tar -xvf prism-4.8-linux64-x86.tar.gz
WORKDIR /root/software/prism-4.8-linux64-x86
RUN bash install.sh
RUN echo "alias prism='bash /root/software/prism-4.8-linux64-x86/bin/prism'" >> ~/.bashrc
RUN echo "alias xprism='bash /root/software/prism-4.8-linux64-x86/bin/xprism'" >> ~/.bashrc


WORKDIR /root/software
RUN git clone https://github.com/tulip-control/tulip-control.git
RUN cd tulip-control && pip3 install -r requirements.txt && pip3 install .

# RUN apt install -qy software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa

# RUN apt-get update

# RUN pip3 install numpy \
# 				stormpy \
# 				seaborn \
# 				networkx==2.0 \
# 				pytest \
# 				numpy \
# 				scipy \
# 				cvxopt \
# 				gr1py

# RUN mkdir software
# WORKDIR /root/software
# RUN wget https://www.prismmodelchecker.org/dl/prism-4.8-linux64-x86.tar.gz
# RUN tar -xvf prism-4.8-linux64-x86.tar.gz
# WORKDIR /root/software/prism-4.8-linux64-x86
# RUN bash install.sh

# RUN echo "alias prism='bash /root/software/prism-4.8-linux64-x86/bin/prism'" >> ~/.bashrc
# RUN echo "alias xprism='bash /root/software/prism-4.8-linux64-x86/bin/xprism'" >> ~/.bashrc
# # Cloning necessary repos i and installing things
# RUN git clone https://github.com/tulip-control/tulip-control.git
# RUN cd tulip-control && pip3 install -r requirements.txt && pip3 install .

# Cloning necessary repos i and installing things
#RUN git clone https://github.com/tulip-control/tulip-control.git


#ENV PATH=$PATH:/home/markov/.local/bin

## User installations
#RUN pip3 install -r tulip-control/requirements.txt