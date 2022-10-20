#!/bin/bash

#######################################################################################
# Builds the required version of zlib.
# Requires common_config.sh in the same directory to run.
# Run it as 
# $ chmod +x ./install_zlib_centos7.sh
# $ sudo ./install_zlib_centos7.sh
#######################################################################################

# make the config file visible regardless of calling location and load it
PATH_PREFIX="${BASH_SOURCE%/*}" 
if [[ ! -d "${PATH_PREFIX}" ]]; then PATH_PREFIX="${PWD}"; fi
. "${PATH_PREFIX}/common_config.sh"

# installation path
ZLIB_INSTALL_PATH=${MUSEN_EXTERNAL_LIBS_PATH}/zlib

# check current version
CURRENT_VERSION=${ZLIB_INSTALL_PATH}/lib/libz.so.${ZLIB_VER}
if [ -f "${CURRENT_VERSION}" ]; then
	echo "zlib " "${CURRENT_VERSION}" " already installed."
	exit 1
fi

# if working directory does not exist yet
if [ ! -d "${MUSEN_WORK_PATH}" ]; then
	# create working directory
	mkdir -p ${MUSEN_WORK_PATH}
	# set proper permissions for current user
	chown ${SUDO_USER}.${SUDO_USER} ${MUSEN_WORK_PATH}
fi
# enter build directory
cd ${MUSEN_WORK_PATH}

# install required libraries
yum -y install wget

# file names
DIR_NAME=zlib-${ZLIB_VER}
TAR_NAME=zlib-${ZLIB_VER}.tar.gz

# clear old
rm -rf ${ZLIB_INSTALL_PATH}

# build zlib
wget http://www.zlib.net/${TAR_NAME}
tar -xvzf ${TAR_NAME}
cd ${DIR_NAME}
# switch to new gcc
source /opt/rh/devtoolset-${GCC_VER}/enable
# build and install
./configure --prefix=${ZLIB_INSTALL_PATH}
make -j$(nproc)
make install

# remove unnecessary 
rm -rf ${ZLIB_INSTALL_PATH}/share
rm -rf ${ZLIB_INSTALL_PATH}/lib/pkgconfig

# clean
cd ../
rm ${TAR_NAME}
rm -rf ${DIR_NAME}
