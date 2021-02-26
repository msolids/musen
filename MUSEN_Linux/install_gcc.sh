#!/bin/bash

#######################################################################################
# Installs the required version of gcc and build tools. Sets the installed version as default.
# Requires common_config.sh in the same derectory to run.
# Run it as 
# $ chmod +x ./install_gcc.sh
# $ ./install_gcc.sh
# Running as 
# $ ./install_gcc.sh
# $ ./install_gcc.sh -l=host
# $ ./install_gcc.sh --location=host
# is equivalent and will perform installation on a local computer.
#######################################################################################

# check for running as root
if [[ ${EUID} -ne 0 ]]; then
   echo "Please run this script as root or using sudo!" 
   exit 1
fi

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
	;;
	
*)
	echo "Error! Unknown location: " "${LOCATION}"
	exit 1
	;;
esac
