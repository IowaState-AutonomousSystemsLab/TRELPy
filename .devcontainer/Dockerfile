FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

RUN echo "hello"
WORKDIR /root

ARG DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true
ENV TZ=America/Chicago

### Setting up all the necessary respositories 
RUN apt-get install -y --reinstall ca-certificates

RUN apt-get update && \
	apt-get -qy install sudo vim git wget curl make build-essential \
	libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev

RUN apt-get update && \
	apt-get install -qy llvm libncurses5-dev \
	libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl python3-distutils

RUN mkdir /root/software
WORKDIR /root/software

ENV PYENV_ROOT "/pyenv"
ENV PATH "$PYENV_ROOT/bin:$PATH"
RUN echo 'eval "$(pyenv init -)" && eval "$(pyenv init --path)"' >> ~/.bashrc
RUN curl https://pyenv.run | bash

RUN eval "$(pyenv init -)" && \
	eval "$(pyenv init --path)" && \
	eval "$(pyenv virtualenv-init -)"


RUN pyenv install 3.10.14 && pyenv global 3.10.14
RUN wget https://bootstrap.pypa.io/get-pip.py 
RUN python3 get-pip.py && rm get-pip.py

# Setting up dependencies
ENV PATH="$PATH:/root/stormpy:/root/carl_storm:/root/storm:/root/pycarl:/root/tulip-control:/root/.local"

RUN sudo apt-get update && apt-get install -qy software-properties-common


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

# Cloning necessary repos i and installing things

RUN wget https://www.prismmodelchecker.org/dl/prism-4.8-linux64-x86.tar.gz
RUN tar -xvf prism-4.8-linux64-x86.tar.gz
WORKDIR /root/software/prism-4.8-linux64-x86
RUN bash install.sh
RUN echo "alias prism='bash /root/software/prism-4.8-linux64-x86/bin/prism'" >> ~/.bashrc
RUN echo "alias xprism='bash /root/software/prism-4.8-linux64-x86/bin/xprism'" >> ~/.bashrc

RUN eval "$(pyenv init -)" && \
	eval "$(pyenv init --path)" && \
	eval "$(pyenv virtualenv-init -)"

## Installing CARL
WORKDIR /root/software
RUN git clone https://github.com/moves-rwth/carl-storm
WORKDIR /root/software/carl-storm
RUN export STORM_DIR=/root/software/carl-storm && mkdir build
WORKDIR /root/software/carl-storm/build
RUN cmake .. && make lib_carl -j8

WORKDIR /root/software
RUN git clone https://github.com/moves-rwth/pycarl.git
WORKDIR /root/software/pycarl
RUN pip install -ve .


RUN pip3 install numpy \
				seaborn \
				networkx==2.0 \
				pytest \
				numpy \
				scipy \
				cvxopt \
				gr1py \
				wheel \
				setuptools \
				pycparser

RUN echo "HOW ARE YOU DOING TODAY"

WORKDIR /root/software
RUN git clone -b master https://github.com/moves-rwth/storm.git
WORKDIR /root/software/storm
RUN export STORM_DIR=/root/software/storm && mkdir build
RUN cd build && cmake .. 
WORKDIR /root/software/storm/build
RUN make storm-main -j8

WORKDIR /root/software
RUN git clone -b master https://github.com/moves-rwth/stormpy.git
WORKDIR /root/software/stormpy
RUN python3 setup.py develop

RUN eval "$(pyenv init -)" && \
	eval "$(pyenv init --path)" && \
	eval "$(pyenv virtualenv-init -)" && \
	pyenv global 3.10.14

RUN pyenv global 3.10.14

# Installing TULIP

WORKDIR /root/software
RUN git clone https://github.com/tulip-control/tulip-control.git
RUN cd /root/software/tulip-control && pip3 install -r requirements.txt
WORKDIR /root/software/tulip-control

RUN eval "$(pyenv init -)" && \
	eval "$(pyenv init --path)" && \
	eval "$(pyenv virtualenv-init -)" && \
	pyenv global 3.10.14

RUN pyenv global 3.10.14

# RUN pip3 install .

RUN apt-get update 

WORKDIR /root/software/
RUN git clone -b main https://github.com/IowaState-AutonomousSystemsLab/TRELPy.git
WORKDIR /root/software/TRELPy

RUN eval "$(pyenv init -)" && \
	eval "$(pyenv init --path)" && \
	eval "$(pyenv virtualenv-init -)" && \
	pyenv global 3.10.14

RUN pyenv global 3.10.14

RUN pip3 install -r requirements-pip.txt
RUN sed 's/#.*//' requirements-apt.txt | xargs sudo apt-get -y install
