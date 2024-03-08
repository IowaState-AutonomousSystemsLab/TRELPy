FROM movesrwth/storm:stable

## Setting up dependencies
# ENV PATH="$PATH:/root/stormpy:/root/carl_storm:/root/storm:/root/pycarl:/root/tulip-control:/root/.local"

RUN apt update && \
    apt install -qy curl python3.8-dev python3.8-distutils && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    cd ~/ && \
    curl https://bootstrap.pypa.io/get-pip.py -o ~/get-pip.py && \
    python get-pip.py

RUN apt-get update && \
	apt-get -y install sudo python3-pip python-is-python3 vim git

RUN apt install -qy software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update

# RUN mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old


# Cloning necessary repos i and installing things
#RUN git clone https://github.com/tulip-control/tulip-control.git

RUN apt-get -y update
RUN apt-get install -y apt-transport-https
RUN apt-get install -y default-jre
RUN apt-get update
RUN apt-get install -y wget \
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
					graphviz

RUN mkdir software
WORKDIR /root/software
RUN wget https://www.prismmodelchecker.org/dl/prism-4.8-linux64-x86.tar.gz
RUN tar -xvf prism-4.8-linux64-x86.tar.gz
WORKDIR /root/software/prism-4.8-linux64-x86
RUN bash install.sh

RUN echo "alias prism='bash /root/software/prism-4.8-linux64-x86/bin/prism'" >> ~/.bashrc
RUN echo "alias xprism='bash /root/software/prism-4.8-linux64-x86/bin/xprism'" >> ~/.bashrc
# Cloning necessary repos i and installing things
RUN git clone https://github.com/tulip-control/tulip-control.git
RUN cd tulip-control && pip3 install -r requirements.txt && pip3 install .


RUN pip3 install numpy \
				stormpy \
				seaborn \
				networkx==2.0 \
				pytest \
				numpy \
				scipy \
				cvxopt \
				gr1py

## Creating the user
#RUN useradd -ms /bin/bash markov -g root -G sudo
#RUN echo 'markov:markov' | chpasswd 
#USER markov
#RUN cd /home/markov
#ENV PATH=$PATH:/home/markov/.local/bin

## User installations
#RUN pip3 install -r tulip-control/requirements.txt