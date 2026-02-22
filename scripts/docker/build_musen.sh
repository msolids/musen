#!/bin/bash

# Copyright (c) 2026, DyssolTEC GmbH. All rights reserved.
# This file is part of MUSEN framework. See LICENSE file for license and warranty information.

set -e

/mnt/musen_src/scripts/copy_to_home.sh
cd ./musen
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DBUILD_CLI=YES \
  -DBUILD_GUI=YES \
  -DINSTALL_AUX_DATA=NO
cmake --build . -- -j$(nproc)
cmake --build . --target install
