#!/bin/bash

#######################################################################################
# Installs the required version of gcc and build tools. 
# Requires common_config.sh in the same derectory to run.
# Run it as 
# $ chmod +x ./install_gcc_centos7.sh
# $ sudo ./install_gcc_centos7.sh
#######################################################################################

# make the config file visible regardless of calling location and load it
PATH_PREFIX="${BASH_SOURCE%/*}" 
if [[ ! -d "${PATH_PREFIX}" ]]; then PATH_PREFIX="${PWD}"; fi
. "${PATH_PREFIX}/common_config.sh"

# check for running as root
if [[ ${EUID} -ne 0 ]]; then
   echo "Please run this script as root or using sudo!" 
   exit 1
fi
	
# check current version
source /opt/rh/devtoolset-${GCC_VER}/enable
CURRENT_VERSION="$(gcc -dumpversion)"
if [ "$(printf '%s\n' "${GCC_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${GCC_VER}" ]; then 
	echo "gcc " ${CURRENT_VERSION} " already installed."
	exit 1
fi

# install build tools
yum install centos-release-scl -y
yum clean all
yum install devtoolset-${GCC_VER}-* -y
# switch to it
source /opt/rh/devtoolset-${GCC_VER}/enable

# print version info
gcc --version
	