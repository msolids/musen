#!/bin/bash

#######################################################################################
# Run
# sudo ./install_prerequisites_local.sh
# to install all required libs and tools. Or run separate scripts as described lower.
# 
# Build requirements:
# gcc - has to be compatible with cuda
# cmake v3.10 or newer - run ./install_cmake_local.sh to build and install into default path
# zlib v1.2.11 - run install_zlib_local.sh script to install into ./MUSEN_externals/zlib_install
# protobuf v3.9.1 - run install_protobuf_local.sh to install into ./MUSEN_externals/protobuf_install
# Qt5 development kit - run install_qt_local.sh script to install into default path
# CUDA v10.1 development kit - run install_cuda_local.sh script to install into default path
#######################################################################################

#######################################################################################
# PATHS TO EXTERNAL LIBRARIES
export Qt_PATH=/usr/lib/x86_64-linux-gnu
export MUSEN_CUDA_PATH=/usr/local/cuda-10.1
export OPENGL_PATH=/usr/lib/x86_64-linux-gnu
#######################################################################################

# update build time
chmod +x linux_update_build_time.sh
./linux_update_build_time.sh

# copy cmake script
cp CMakeLists.txt MUSEN_src/

# png fix
#find . -type f -iname '*.png' -exec pngcrush -ow -rem allb -reduce {} \; > png_cleanup_log 2>&1

# cuda
PATH="$PATH":${MUSEN_CUDA_PATH}/bin/

# protobuf
export MUSEN_PROTO_PATH=$PWD/MUSEN_externals/protobuf_install
PATH="$PATH":${MUSEN_PROTO_PATH}/bin/
LD_LIBRARY_PATH="$LD_LIBRARY_PATH":${MUSEN_PROTO_PATH}/lib/
export CPLUS_INCLUDE_PATH=${MUSEN_PROTO_PATH}/include/ 

# zlib
export MUSEN_ZLIB_PATH=$PWD/MUSEN_externals/zlib_install

#######################################################################################

if [ ! -d MUSEN_build ]; then
	mkdir MUSEN_build
fi
cd MUSEN_build
cmake protobuf ../MUSEN_src
make protobuf -j 7
cmake ../MUSEN_src
make -j 7
cd ../
