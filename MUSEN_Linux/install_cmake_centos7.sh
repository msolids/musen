#!/bin/bash

#######################################################################################
# Installs the required version of cmake and removes the previously installed ones.
# Requires common_config.sh in the same derectory to run.
# Run it as 
# $ chmod +x ./install_cmake_centos7.sh
# $ sudo ./install_cmake_centos7.sh
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

# check current version
CURRENT_VERSION="$(cmake3 --version | head -n1 | cut -d' ' -f3)"
if [ "$(printf '%s\n' "${CMAKE_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${CMAKE_VER}" ]; then 
	echo "cmake " ${CURRENT_VERSION} " already installed."
	exit 1
fi

# install cmake
yum -y install cmake3

# print version info
cmake3 --version

# check current version
# CURRENT_VERSION="$(cmake --version | head -n1 | cut -d' ' -f3)"
# if [ "$(printf '%s\n' "${CMAKE_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${CMAKE_VER}" ]; then 
	# echo "cmake " ${CURRENT_VERSION} " already installed."
	# exit 1
# fi

# uninstall previous version
# yum -y remove cmake
# yum -y remove cmake-data
	
# install required libraries
# yum -y install wget
# yum -y install openssl-devel

# remove  previous downloads
# rm cmake-${CMAKE_VER}.tar.gz
# rm -r temp-build-cmake-${CMAKE_VER}
# download sources
# wget https://cmake.org/files/v${CMAKE_VER_MAJOR}/cmake-${CMAKE_VER}.tar.gz
# unzip
# mkdir temp-build-cmake-${CMAKE_VER}
# tar zxvf cmake-${CMAKE_VER}.tar.gz --directory ./temp-build-cmake-${CMAKE_VER}/
# cd temp-build-cmake-${CMAKE_VER}/cmake-${CMAKE_VER}
# switch to new gcc
# source /opt/rh/devtoolset-${GCC_VER}/enable
# build and install
# ./bootstrap --prefix=/usr/local
# make -j$(nproc)
# make install

# print version info
# cmake --version
	
	
