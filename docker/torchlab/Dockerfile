# Nvidia provides a convenient Docker image for GPU use
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# This should get python 3.8
RUN apt-get update && apt-get install -y python3 python3-pip

ADD requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install jupyterlab ipywidgets
WORKDIR /torchlab/
