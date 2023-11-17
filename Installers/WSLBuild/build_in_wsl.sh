#!/bin/bash

# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

#######################################################################################
# This script builds MUSEN in WSL.
# Since file operations in Windows directories in WSL are slow, 
# all files are copied to WSL home directory and built there.
#######################################################################################

solution_dir=$(wslpath "$1")
results_path=$(wslpath "$2")
targets_params=$3
wsl_src_path="${HOME}/musen_wsl"
wsl_build_path="${wsl_src_path}/build"
wsl_install_path="${wsl_src_path}/install"

echo "Source: ${solution_dir}"
echo "Destination: ${results_path}"
echo "${targets_params}"

rm -rf ${wsl_build_path}
rm -rf ${wsl_install_path}
mkdir -p ${wsl_build_path}
mkdir -p ${wsl_install_path}
rsync -av --exclude=.vs --exclude=build --exclude=ExternalLibraries --exclude=install --exclude=Installers/Compiler --exclude=Installers/Installers --exclude=MUSEN_Linux --exclude=x64 ${solution_dir} ${wsl_src_path}
cd ${wsl_build_path}
cmake .. -DCMAKE_INSTALL_PREFIX=${wsl_install_path} ${targets_params}
cmake --build . --parallel $(nproc)
make install
rsync -av ${wsl_install_path}/ ${results_path}/
