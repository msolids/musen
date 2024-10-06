#!/bin/bash

# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

#######################################################################################
# This script downloads and builds the required version of protobuf.
# To build MUSEN with the built library, override paths to protobuf as
# export LD_LIBRARY_PATH=${PWD}/protobuf/lib:$LD_LIBRARY_PATH && cmake .. -DCMAKE_INSTALL_PREFIX=../install -DProtobuf_INCLUDE_DIR=${PWD}/protobuf/include -DProtobuf_LIBRARY=${PWD}/protobuf/lib/libprotobuf.so -DProtobuf_PROTOC_EXECUTABLE=${PWD}/protobuf/bin/protoc
#######################################################################################

PROTO_VER=3.14.0

# installation path
PROTO_INSTALL_PATH=${PWD}/protobuf

# path to zlib
ZLIB_INSTALL_PATH=${PWD}/zlib

# file names
DIR_NAME=protobuf-${PROTO_VER}
TAR_NAME=protobuf-cpp-${PROTO_VER}.tar.gz

# clear old installation directory
rm -rf ${PROTO_INSTALL_PATH}

# download 
wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTO_VER}/${TAR_NAME}
# extract
tar -xvzf ${TAR_NAME}

# build and install
cd ${DIR_NAME}
./configure --with-zlib --with-zlib-include=${ZLIB_INSTALL_PATH}/include --with-zlib-lib=${ZLIB_INSTALL_PATH}/lib --disable-examples --disable-tests --prefix=${PROTO_INSTALL_PATH}
make -j 8
make install

# clean
cd ../
rm ${TAR_NAME}
rm -rf ${DIR_NAME}
