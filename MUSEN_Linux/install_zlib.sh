#!/bin/bash

#######################################################################################
# Builds the required version of zlib.
# Requires common_config.sh in the same directory to run.
# Run it as 
# $ chmod +x ./install_zlib.sh
# $ ./install_zlib.sh
# Running as 
# $ ./install_zlib.sh
# $ ./install_zlib.sh -l=host
# $ ./install_zlib.sh --location=host
# is equivalent and will perform installation on a local computer.
# Run as 
# $ ./install_zlib.sh -l=hpc5
# $ ./install_zlib.sh --location=hpc5
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

	# installation path
	ZLIB_INSTALL_PATH=${MUSEN_EXTERNAL_LIBS_PATH}/zlib

	# if working directory does not exist yet
	if [ ! -d "${MUSEN_WORK_PATH}" ]; then
		# create working directory
		mkdir -p ${MUSEN_WORK_PATH}
		# set proper permissions for current user
		chown ${SUDO_USER}.${SUDO_USER} ${MUSEN_WORK_PATH}
	fi
	# enter build directory
	cd ${MUSEN_WORK_PATH}
	
	;;
	
hpc5)

	# installation path
	ZLIB_INSTALL_PATH=${PWD}/${MUSEN_EXTERNAL_LIBS_DIR}/zlib

	# load proper gcc
	. /etc/bashrc
	module load gcc/9.2.0
	
	;;
	
*)
	echo "Error! Unknown location: " "${LOCATION}"
	exit 1
	;;
esac

# check current version
CURRENT_VERSION=${ZLIB_INSTALL_PATH}/lib/libz.so.${ZLIB_VER}
if [ -f "${CURRENT_VERSION}" ]; then
	echo "zlib " "${CURRENT_VERSION}" " already installed."
	exit 1
fi

# file names
DIR_NAME=zlib-${ZLIB_VER}
TAR_NAME=zlib-${ZLIB_VER}.tar.gz

# clear old
rm -rf ${ZLIB_INSTALL_PATH}

# build zlib
wget http://www.zlib.net/${TAR_NAME}
tar -xvzf ${TAR_NAME}
cd ${DIR_NAME}
./configure --prefix=${ZLIB_INSTALL_PATH}
make -j 8
make install

# remove unnecessary 
rm -rf ${ZLIB_INSTALL_PATH}/share
rm -rf ${ZLIB_INSTALL_PATH}/lib/pkgconfig

# clean
cd ../
rm ${TAR_NAME}
rm -rf ${DIR_NAME}
