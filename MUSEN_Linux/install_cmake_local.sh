#!/bin/bash

# version to compile
CMAKE_VER=3.14
CMAKE_BUILD=5
CMAKE_FULL_VER=$CMAKE_VER.$CMAKE_BUILD

# check current version
CURRENT_VERSION="$(cmake --version | head -n1 | cut -d' ' -f3)"
if [ "$(printf '%s\n' "$CMAKE_FULL_VER" "$CURRENT_VERSION" | sort -V | head -n1)" = "$CMAKE_FULL_VER" ]; then 
	exit 1
fi

# uninstall previous version
apt -y remove --purge --auto-remove cmake

# download and extract
mkdir cmake_build_temp
cd cmake_build_temp
wget https://cmake.org/files/v$CMAKE_VER/cmake-$CMAKE_FULL_VER.tar.gz
tar -xzvf cmake-$CMAKE_FULL_VER.tar.gz
cd cmake-$CMAKE_FULL_VER/

# build and install
./bootstrap
make -j8
make install

# clean
cd ../../
rm -rf cmake_build_temp
