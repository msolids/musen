#!/bin/bash

#######################################################################################
# Compiles MUSEN locally.
# Requires common_config.sh in the same directory to run.
#
# To select version of MUSEN to build, run the script as
# $ ./make_musen_host.sh [--target=cli] [--target=gui] [--target=matlab]
# or 
# $ ./make_musen_host.sh [-t=c] [-t=g] [-t=m]
# where 
# cli/c    - command line version   (cmusen)
# gui/g    - GUI version            (musen)
# matlab/m - matlab library version (mmusen)
# Running the script without target parameters will build all versions. 
# Running the script with at least one target will disable the rest not mentioned targets.
#
# Before running the script for the first time, run
# $ sudo ./install_prerequisites_host.sh
# to install all required libs and tools. Or run separate scripts as described further.
# 
# Build requirements:
# gcc 10               - run './install_gcc.sh      -l=host' to install into default path
# cmake v3.18 or newer - run './install_cmake.sh    -l=host' to install into default path
# zlib v1.2.11         - run './install_zlib.sh     -l=host' to install into ${MUSEN_EXTERNAL_LIBS_PATH}/zlib
# protobuf v3.14.0     - run './install_protobuf.sh -l=host' to install into ${MUSEN_EXTERNAL_LIBS_PATH}/protobuf
# Qt v5.15.2           - run './install_qt.sh       -l=host' to install into ${MUSEN_EXTERNAL_LIBS_PATH}/qt
# CUDA v11.2           - run './install_cuda.sh     -l=host' to install into default path
# MATLAB 2019b (optional)
#######################################################################################

# make the config file visible regardless of calling location and load it
PATH_PREFIX="${BASH_SOURCE%/*}" 
if [[ ! -d "${PATH_PREFIX}" ]]; then PATH_PREFIX="${PWD}"; fi
. "${PATH_PREFIX}/common_config.sh"

# parse arguments
BUILD_CLI=no
BUILD_GUI=no
BUILD_MAT=no
for i in "$@"
do
case $i in
-t=*|--target=*)
	TARGET="${i#*=}"
	case $TARGET in
	c|cli)
		BUILD_CLI=yes
		;;
	g|gui)
		BUILD_GUI=yes
		;;
	m|matlab)
		BUILD_MAT=yes
		;;
	*)
		echo "Error! Unknown target: " "${TARGET}"
		exit 1
		;;
	esac
	shift
	;;
*)
	echo "Error! Unknown option: " "${i}"
	exit 1
	;;
esac
done
# if no targets defined explicitly, build all
if [[ ${BUILD_CLI} == "no" && ${BUILD_GUI} == "no" && ${BUILD_MAT} == "no" ]]; then
	BUILD_CLI=yes
	BUILD_GUI=yes
	BUILD_MAT=yes
fi

#######################################################################################
# PATHS TO EXTERNAL LIBRARIES
export MUSEN_ZLIB_PATH=${MUSEN_EXTERNAL_LIBS_PATH}/zlib
export MUSEN_PROTO_PATH=${MUSEN_EXTERNAL_LIBS_PATH}/protobuf
export MUSEN_QT_PATH=${MUSEN_EXTERNAL_LIBS_PATH}/qt
export MUSEN_CUDA_PATH=/usr/local/cuda-${CUDA_VER}
export OPENGL_PATH=/usr/lib/x86_64-linux-gnu
#######################################################################################

# update build time
./generate_time_header.sh
mv -f ./BuildTime.h ${MUSEN_SRC_PATH}/Version/

# copy cmake script
cp CMakeLists.txt ${MUSEN_SRC_PATH}/

# png fix
#find ${MUSEN_SRC_PATH}/ -type f -iname '*.png' -exec pngcrush -ow -rem allb -reduce {} \; > png_cleanup_log 2>&1

# cuda
PATH="$PATH":${MUSEN_CUDA_PATH}/bin/

# protobuf
PATH="$PATH":${MUSEN_PROTO_PATH}/bin/
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}":${MUSEN_PROTO_PATH}/lib/
export CPLUS_INCLUDE_PATH=${MUSEN_PROTO_PATH}/include/ 

# qt
export CMAKE_PREFIX_PATH=${MUSEN_QT_PATH}/${QT_VER}/gcc_64:${CMAKE_PREFIX_PATH}

# create build directory 
if [ ! -d ${MUSEN_BUILD_PATH} ]; then
	mkdir ${MUSEN_BUILD_PATH}
fi

#######################################################################################
# start build

cmake protobuf -S ${MUSEN_SRC_PATH} -B ${MUSEN_BUILD_PATH}
make protobuf --directory ${MUSEN_BUILD_PATH} --silent --jobs=8
cmake -S ${MUSEN_SRC_PATH} -B ${MUSEN_BUILD_PATH}
if [[ ${BUILD_CLI} == "yes" ]]; then
	make cmusen  --directory ${MUSEN_BUILD_PATH} --keep-going --silent --jobs=8
fi
if [[ ${BUILD_GUI} == "yes" ]]; then
	make musen   --directory ${MUSEN_BUILD_PATH} --keep-going --silent --jobs=8
fi
if [[ ${BUILD_MAT} == "yes" ]]; then
	make mmusen  --directory ${MUSEN_BUILD_PATH} --keep-going --silent --jobs=8
fi
