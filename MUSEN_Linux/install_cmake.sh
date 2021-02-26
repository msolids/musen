#!/bin/bash

#######################################################################################
# Installs the required version of cmake and removes the installed ones.
# Requires common_config.sh in the same derectory to run.
# Run it as 
# $ chmod +x ./install_cmake.sh
# $ ./install_cmake.sh
# Running as 
# $ ./install_cmake.sh
# $ ./install_cmake.sh -l=host
# $ ./install_cmake.sh --location=host
# is equivalent and will perform installation on a local computer.
# Run as 
# $ ./install_cmake.sh -l=hpc5
# $ ./install_cmake.sh --location=hpc5
# to perform installation on HPC5.
#######################################################################################

# make the config file visible regardless of calling location and load it
PATH_PREFIX="${BASH_SOURCE%/*}" 
if [[ ! -d "${PATH_PREFIX}" ]]; then PATH_PREFIX="${PWD}"; fi
. "${PATH_PREFIX}/common_config.sh"

# parse arguments
LOCATION=host
for i in "$@"
do
case $i in
-l=*|--location=*)
	LOCATION="${i#*=}"
	shift
	;;
*)
	echo "Error! Unknown option: " "${i}"
	;;
esac
done

case ${LOCATION} in
host)

	# check for running as root
	if [[ ${EUID} -ne 0 ]]; then
	   echo "Please run this script as root or using sudo!" 
	   exit 1
	fi

	# check current version
	CURRENT_VERSION="$(cmake --version | head -n1 | cut -d' ' -f3)"
	if [ "$(printf '%s\n' "${CMAKE_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${CMAKE_VER}" ]; then 
		echo "cmake " ${CURRENT_VERSION} " already installed."
		exit 1
	fi

	# uninstall previous version
	apt -y autoremove --purge cmake
	apt -y autoremove --purge cmake-data

	# install required packages
	apt-get update
	apt -y install apt-transport-https ca-certificates gnupg software-properties-common wget

	# obtain a copy of signing key:
	wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

	# add repo
	UBUNTU_CODENAME="$(lsb_release -c | cut -d$'\t' -f2)"
	apt-add-repository "deb https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main"
	apt update

	# install additional package
	apt -y install kitware-archive-keyring
	rm /etc/apt/trusted.gpg.d/kitware.gpg

	# install cmake
	apt -y install cmake
	
	# if another gcc is installed with cmake, add it to alternatives list
	CURRENT_GCC_VERSION="$(gcc -dumpversion)"
	if [[ ${CURRENT_GCC_VERSION} ]]; then
		update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${CURRENT_GCC_VERSION} ${CURRENT_GCC_VERSION} --slave /usr/bin/g++ g++ /usr/bin/g++-${CURRENT_GCC_VERSION} --slave /usr/bin/gcov gcov /usr/bin/gcov-${CURRENT_GCC_VERSION}
	fi
	
	;;
	
hpc5)

	# installation path
	INSTALL_PATH=${PWD}/${MUSEN_EXTERNAL_LIBS_DIR}
	CMAKE_INSTALL_PATH=${INSTALL_PATH}/cmake

	# check current version
	CURRENT_VERSION="$(${CMAKE_INSTALL_PATH}/bin/cmake --version | head -n1 | cut -d' ' -f3)"
	if [ "$(printf '%s\n' "${CMAKE_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${CMAKE_VER}" ]; then 
		echo "cmake " ${CURRENT_VERSION} " already installed."
		exit 1
	fi
	
	# file names
	DIR_NAME=cmake-${CMAKE_VER}-Linux-x86_64
	TAR_NAME=${DIR_NAME}.tar.gz

	# create path
	mkdir ${INSTALL_PATH}

	# download and extract
	wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/${TAR_NAME}
	tar -C ${INSTALL_PATH} -xzvf ${TAR_NAME}

	# rename
	mv ${INSTALL_PATH}/${DIR_NAME} ${CMAKE_INSTALL_PATH}

	# clean
	rm ${TAR_NAME}
	
	;;
	
*)
	echo "Error! Unknown location: " "${LOCATION}"
	exit 1
	;;
esac
