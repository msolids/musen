#!/bin/bash

#######################################################################################
# Installs the required version of CUDA and removes the installed ones. 
# Requires common_config.sh in the same directory to run.
# Run it as 
# $ chmod +x ./install_cuda_centos7.sh
# $ sudo ./install_cuda_centos7.sh
#######################################################################################

# make the config file visible regardless of calling location and load it
PATH_PREFIX="${BASH_SOURCE%/*}" 
if [[ ! -d "${PATH_PREFIX}" ]]; then PATH_PREFIX="${PWD}"; fi
. "${PATH_PREFIX}/common_config.sh"

# check for running as root
if [[ ${EUID} -ne 0 ]]; then
   echo "Please run this script as root or using sudo!" 
   exit 1
fi

# get information about distro
# DISTRO_NAME=$(lsb_release -i | cut -d$'\t' -f2 | cut -d' ' -f2 | tr '[:upper:]' '[:lower:]')
# DISTRO_VERSION=$(lsb_release -r | cut -d$'\t' -f2 | tr -d .)
# DISTRO=${DISTRO_NAME}${DISTRO_VERSION}
# ARCH=$(uname -m)
# KERNEL_RELEASE=$(uname -r)

# check current version
CURRENT_VERSION="$(/usr/local/cuda-${CUDA_VER}/bin/nvcc --version | tail -n2 | head -n1 | cut -d',' -f2 | cut -d' ' -f3)"
if [ "$(printf '%s\n' "${CUDA_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${CUDA_VER}" ]; then 
	echo "cuda " ${CURRENT_VERSION} " already installed."
	exit 1
fi

# install cuda toolkit
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum clean all
yum -y install cuda-toolkit-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR}

# post-install actions
export PATH=/usr/local/cuda-${CUDA_VER_MAJOR}.${CUDA_VER_MINOR}/bin${PATH:+:${PATH}}

# print version info
nvcc --version

# try to uninstall previous versions
# /usr/local/cuda-10.0/bin/cuda-uninstaller
# /usr/local/cuda-10.1/bin/cuda-uninstaller
# /usr/local/cuda-10.2/bin/cuda-uninstaller
# /usr/local/cuda-11.0/bin/cuda-uninstaller
# /usr/local/cuda-11.1/bin/cuda-uninstaller
# /usr/bin/nvidia-uninstall
# yum remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"

# install required libraries
# yum install subscription-manager

# enable EPEL
# yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
# enable optional repositories.
# subscription-manager repos --enable=rhel-7-workstation-optional-rpms

# install repository meta-data
# yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
# clean yum repository cache
# yum clean expire-cache

# install cuda
# yum -y install cuda

# post-install actions
# export PATH=/usr/local/cuda-${CUDA_VER_MAJOR}.${CUDA_VER_MINOR}/bin${PATH:+:${PATH}}
	