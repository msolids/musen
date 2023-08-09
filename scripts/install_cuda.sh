#!/bin/bash

# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

#######################################################################################
# This script installs CUDA toolkit of the specified version 
# following the official NVIDIA guide for WSL and Ubuntu:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 
#######################################################################################

CUDA_VER_MAJOR=11
CUDA_VER_MINOR=7
CUDA_VER=${CUDA_VER_MAJOR}.${CUDA_VER_MINOR}

# get information about distro
DISTRO_NAME=$(lsb_release -i | cut -d$'\t' -f2 | cut -d' ' -f2 | tr '[:upper:]' '[:lower:]')
DISTRO_VERSION=$(lsb_release -r | cut -d$'\t' -f2 | tr -d .)
DISTRO=${DISTRO_NAME}${DISTRO_VERSION}
ARCH=$(uname -m)
KERNEL_RELEASE=$(uname -r)

# remove CUDA toolkit
# sudo apt --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
# sudo apt autoremove

# remove outdated signing key
sudo apt-key del 7fa2af80

if [[ "${KERNEL_RELEASE}" == *"WSL"* ]]; then # WSL
	# download the newcuda-keyring package
	wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
	
else # Ununtu
	# install kernel headers and development packages for the currently running kernel
	sudo apt install linux-headers-${KERNEL_RELEASE}
		
	# install the newcuda-keyring package
	wget https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb
fi

# install the newcuda-keyring package
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb

# update apt repository
sudo apt update

# install cuda
sudo apt -y install cuda-toolkit-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR}

# post-install actions
export PATH=/usr/local/cuda-${CUDA_VER_MAJOR}.${CUDA_VER_MINOR}/bin${PATH:+:${PATH}}
echo "export PATH=/usr/local/cuda-${CUDA_VER_MAJOR}.${CUDA_VER_MINOR}/bin"'${PATH:+:${PATH}}' >> ~/.bashrc
