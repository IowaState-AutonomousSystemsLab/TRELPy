FROM movesrwth/storm:stable

## Setting up dependencies
# ENV PATH="$PATH:/root/stormpy:/root/carl_storm:/root/storm:/root/pycarl:/root/tulip-control:/root/.local"
RUN apt-get update && \
	apt-get -y install sudo python3-pip python-is-python3 vim git

RUN mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old


# Cloning necessary repos i and installing things
#RUN git clone https://github.com/tulip-control/tulip-control.git

RUN pip3 install numpy
RUN pip3 install stormpy
RUN pip3 install tulip
RUN pip3 install seaborn

RUN apt-get install -y apt-transport-https
RUN apt-get -y update
RUN apt-get install -y default-jre
RUN apt-get update
RUN apt-get install -y wget default-jdk

RUN mkdir software
WORKDIR /root/software
RUN wget https://www.prismmodelchecker.org/dl/prism-4.8-linux64-x86.tar.gz
RUN tar -xvf prism-4.8-linux64-x86.tar.gz
WORKDIR /root/software/prism-4.8-linux64-x86
RUN bash install.sh

RUN echo "alias prism='bash /root/software/prism-4.8-linux64-x86/bin/prism'" >> ~/.bashrc
RUN echo "alias xprism='bash /root/software/prism-4.8-linux64-x86/bin/xprism'" >> ~/.bashrc

## Creating the user
#RUN useradd -ms /bin/bash markov -g root -G sudo
#RUN echo 'markov:markov' | chpasswd 
#USER markov
#RUN cd /home/markov
#ENV PATH=$PATH:/home/markov/.local/bin

## User installations
#RUN pip3 install -r tulip-control/requirements.txt