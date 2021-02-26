#!/bin/bash

#######################################################################################
# Builds the required version of protobuf.
# Requires common_config.sh in the same directory to run.
# Run it as 
# $ chmod +x ./install_protobuf.sh
# $ ./install_protobuf.sh
# Running as 
# $ ./install_protobuf.sh
# $ ./install_protobuf.sh -l=host
# $ ./install_protobuf.sh --location=host
# is equivalent and will perform installation on a local computer.
# Run as 
# $ ./install_protobuf.sh -l=hpc5
# $ ./install_protobuf.sh --location=hpc5
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
	PROTO_INSTALL_PATH=${MUSEN_EXTERNAL_LIBS_PATH}/protobuf

	# path to zlib
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
	PROTO_INSTALL_PATH=${PWD}/${MUSEN_EXTERNAL_LIBS_DIR}/protobuf

	# path to zlib
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
CURRENT_VERSION="$(${PROTO_INSTALL_PATH}/bin/protoc --version | head -n1 | cut -d' ' -f2)"
if [ "$(printf '%s\n' "${PROTO_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${PROTO_VER}" ]; then 
	echo "protobuf " ${CURRENT_VERSION} " already installed."
	exit 1
fi

# file names
DIR_NAME=protobuf-${PROTO_VER}
TAR_NAME=protobuf-cpp-${PROTO_VER}.tar.gz

# clear old
rm -rf ${PROTO_INSTALL_PATH}

# build protobuf
wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTO_VER}/${TAR_NAME}
tar -xvzf ${TAR_NAME}
cd ${DIR_NAME}
./configure --with-zlib --with-zlib-include=${ZLIB_INSTALL_PATH}/include --with-zlib-lib=${ZLIB_INSTALL_PATH}/lib --disable-examples --disable-tests --prefix=${PROTO_INSTALL_PATH}
make -j 8
make install

# remove unnecessary 
rm -rf ${PROTO_INSTALL_PATH}/lib/pkgconfig
find ${PROTO_INSTALL_PATH}/lib/ -name "libprotobuf-lite.*" -exec rm -rf {} \;

# clean
cd ../
rm ${TAR_NAME}
rm -rf ${DIR_NAME}
