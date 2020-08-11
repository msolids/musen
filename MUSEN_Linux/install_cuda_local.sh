#!/bin/bash

# Installs CUDA 10.1 for Ubuntu 16.04 / 18.04

# version to install
CUDA_VER=10.1

# file name
RUN_NAME=cuda_$CUDA_VER.168_418.67_linux.run

# download package
wget https://developer.nvidia.com/compute/cuda/$CUDA_VER/Prod/local_installers/$RUN_NAME

# make executable
chmod +x $RUN_NAME

# install
./$RUN_NAME --silent --toolkit

# clean
rm $RUN_NAME