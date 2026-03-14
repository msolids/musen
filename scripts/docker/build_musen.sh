#!/bin/bash

# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
# See LICENSE file for license and warranty information.

# Build MUSEN in a Docker container.
# Assumes that the source code is mounted to /mnt/musen_src 
# and that the container has necessary build tools installed.

# Exit immediately if a command exits with a non-zero status
set -o errexit

/mnt/musen_src/scripts/copy_to_home.sh
cd ~/musen
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DMUSEN_BUILD_CLI=ON \
  -DMUSEN_BUILD_GUI=ON \
  -DMUSEN_INSTALL_DATA=OFF
cmake --build . --parallel $(nproc)
cmake --install .
