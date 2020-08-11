#!/bin/bash

#######################################################################################
# Will install all packages and compile some libraries needed to build MUSEN.
# Run it as 
# dos2unix ./install_prerequisites_local.sh
# sudo chmod +x ./install_prerequisites_local.sh
# sudo ./install_prerequisites_local.sh
#######################################################################################

#install gcc compiler, libs and tools
apt -y install gcc
apt -y install build-essential
# cmake
dos2unix ./install_cmake_local.sh
chmod +x ./install_cmake_local.sh
./install_cmake_local.sh
# zlib needed to build protobuf
dos2unix ./install_zlib_local.sh
chmod +x ./install_zlib_local.sh
./install_zlib_local.sh
# protobuf
dos2unix ./install_protobuf_local.sh
chmod +x ./install_protobuf_local.sh
./install_protobuf_local.sh
# qt
dos2unix ./install_qt_local.sh
chmod +x ./install_qt_local.sh
./install_qt_local.sh
# libs for qt
apt -y install libfontconfig1
apt -y install mesa-common-dev
apt -y install libgl1-mesa-dev
# cuda
dos2unix ./install_cuda_local.sh
chmod +x ./install_cuda_local.sh
./install_cuda_local.sh
