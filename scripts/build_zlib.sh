#!/bin/bash

# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

#######################################################################################
# This script downloads and builds the required version of zlib.
# It is needed to build protobuf (see build_protobuf.sh)
#######################################################################################

ZLIB_VER=1.2.13

# installation path
ZLIB_INSTALL_PATH=${PWD}/zlib

# file names
DIR_NAME=zlib-${ZLIB_VER}
TAR_NAME=v${ZLIB_VER}.tar.gz

# clear old installation directory
rm -rf ${ZLIB_INSTALL_PATH}

# download 
wget https://github.com/madler/zlib/archive/refs/tags/${TAR_NAME}
# extract
tar -xvzf ${TAR_NAME}

# build and install
cd ${DIR_NAME}
./configure --prefix=${ZLIB_INSTALL_PATH}
make -j 8
make install

# clean
cd ../
rm ${TAR_NAME}
rm -rf ${DIR_NAME}
