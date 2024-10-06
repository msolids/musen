#!/bin/bash

# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

#######################################################################################
# Installs the selected version of gcc and build tools. Sets the installed version as default.
#######################################################################################

GCC_VER=11.4.0

# check current version
CURRENT_VERSION="$(gcc -dumpversion)"
if [ "$(printf '%s\n' "${GCC_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${GCC_VER}" ]; then 
	echo "gcc " ${CURRENT_VERSION} " already installed."
	exit 1
fi

# install additional tools
# apt -y install build-essential

# add repository
apt -y install software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt update

# install gcc
apt -y install gcc-${GCC_VER}
apt -y install g++-${GCC_VER}

# set installed version as default. use 'sudo update-alternatives --config gcc' to choose another version
if [[ ${CURRENT_VERSION} ]]; then
	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${CURRENT_VERSION} ${CURRENT_VERSION} --slave /usr/bin/g++ g++ /usr/bin/g++-${CURRENT_VERSION} --slave /usr/bin/gcov gcov /usr/bin/gcov-${CURRENT_VERSION}
fi
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VER} ${GCC_VER} --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VER} --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VER}
update-alternatives --display gcc
