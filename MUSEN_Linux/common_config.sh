#!/bin/bash

# directory name where third-party libraries are installed
MUSEN_EXTERNAL_LIBS_DIR="external_libs"
# directory name where source files are gathered
MUSEN_SRC_DIR="src"
# directory name where build takes place
MUSEN_BUILD_DIR="build"
# directory name where compiled files are placed
MUSEN_RESULTS_DIR="compiled"

# real path to user home directory - works even if the script is called with sudo 
REAL_USER="${SUDO_USER:-$USER}"
REAL_USER_HOME_DIR=$( getent passwd "${REAL_USER}" | cut -d: -f6 )
# path to directory where compilation takes place
MUSEN_WORK_PATH="${REAL_USER_HOME_DIR}/MUSEN"
# path to directory where third-party libraries are installed
MUSEN_EXTERNAL_LIBS_PATH="${MUSEN_WORK_PATH}/${MUSEN_EXTERNAL_LIBS_DIR}"
# path to directory where source files are gathered
MUSEN_SRC_PATH="${MUSEN_WORK_PATH}/${MUSEN_SRC_DIR}"
# path to directory where build takes place
MUSEN_BUILD_PATH="${MUSEN_WORK_PATH}/${MUSEN_BUILD_DIR}"

# path to directory where build takes place
MUSEN_RESULTS_PATH="${PWD}/${MUSEN_RESULTS_DIR}"

### versions of tools and libs
# gcc
GCC_VER=9
# cmake
CMAKE_VER=3.18.0
# zlib
ZLIB_VER=1.2.12
# protobuf
PROTO_VER=3.14.0
# qt
QT_VER_MAJOR=5
QT_VER_MINOR=15
QT_VER_BUILD=2
QT_VER=${QT_VER_MAJOR}.${QT_VER_MINOR}.${QT_VER_BUILD}
# qt installer
QT_INSTALLER_VER_MAJOR=4
QT_INSTALLER_VER_MINOR=0
QT_INSTALLER_VER_BUILD=1
QT_INSTALLER_VER=${QT_INSTALLER_VER_MAJOR}.${QT_INSTALLER_VER_MINOR}.${QT_INSTALLER_VER_BUILD}
# cuda
CUDA_VER_MAJOR=11
CUDA_VER_MINOR=2
CUDA_VER=${CUDA_VER_MAJOR}.${CUDA_VER_MINOR}
