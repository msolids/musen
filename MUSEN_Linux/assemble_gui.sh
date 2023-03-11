#!/bin/bash

#######################################################################################
# Creates an archive with all files needed for GUI version, including Qt libs.
# Requires common_config.sh in the same directory to run.
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

# select paths depending on the selected location
case ${LOCATION} in
host)
	ACTIVE_WORK_PATH=${MUSEN_WORK_PATH}
	ACTIVE_BUILD_PATH=${MUSEN_BUILD_PATH}
	ACTIVE_SRC_PATH=${MUSEN_SRC_PATH}
	ACTIVE_EXTERNAL_LIBS_PATH=${MUSEN_EXTERNAL_LIBS_PATH}
	;;
	
hpc5)
	ACTIVE_WORK_PATH=$(eval "echo ${MUSEN_HPC5_WORK_PATH}")
	ACTIVE_BUILD_PATH=$(eval "echo ${MUSEN_HPC5_BUILD_PATH}")
	ACTIVE_SRC_PATH=$(eval "echo ${MUSEN_HPC5_SRC_PATH}")
	ACTIVE_EXTERNAL_LIBS_PATH=$(eval "echo ${MUSEN_HPC5_EXTERNAL_LIBS_PATH}")
	;;
	
*)
	echo "Error! Unknown location: " "${LOCATION}"
	exit 1
	;;
esac

# check if binary file exist
if [ ! -f "${ACTIVE_BUILD_PATH}/musen" ]; then
    exit 1
fi

# a directory for assembly 
ASSEMBLY_DIR="musen_gui"
ASSEMBLY_PATH=${ACTIVE_WORK_PATH}/${ASSEMBLY_DIR}

# create the directory for assembly 
mkdir -p ${ASSEMBLY_PATH}

# create all directories
mkdir -p ${ASSEMBLY_PATH}/bin
mkdir -p ${ASSEMBLY_PATH}/lib
mkdir -p ${ASSEMBLY_PATH}/plugins/imageformats
mkdir -p ${ASSEMBLY_PATH}/plugins/platforms
mkdir -p ${ASSEMBLY_PATH}/plugins/xcbglintegrations
mkdir -p ${ASSEMBLY_PATH}/styles

# move compiled binary to the appropriate directory
rsync --info=progress2 -az ${ACTIVE_BUILD_PATH}/musen ${ASSEMBLY_PATH}/bin/

# copy startup script
rsync --info=progress2 -az ${ACTIVE_WORK_PATH}/musen.sh ${ASSEMBLY_PATH}/

# copy qt libraries
rsync --info=progress2 --copy-links -az ${ACTIVE_EXTERNAL_LIBS_PATH}/qt/${QT_VER}/gcc_64/lib/{libicudata.so.56,libicui18n.so.56,libicuuc.so.56,libQt5Core.so.5,libQt5DBus.so.5,libQt5Gui.so.5,libQt5OpenGL.so.5,libQt5Widgets.so.5,libQt5XcbQpa.so.5} ${ASSEMBLY_PATH}/lib/
rsync --info=progress2 --copy-links -az ${ACTIVE_EXTERNAL_LIBS_PATH}/qt/${QT_VER}/gcc_64/plugins/imageformats/libqjpeg.so ${ASSEMBLY_PATH}/plugins/imageformats/
rsync --info=progress2 --copy-links -az ${ACTIVE_EXTERNAL_LIBS_PATH}/qt/${QT_VER}/gcc_64/plugins/platforms/libqxcb.so ${ASSEMBLY_PATH}/plugins/platforms/
rsync --info=progress2 --copy-links -az ${ACTIVE_EXTERNAL_LIBS_PATH}/qt/${QT_VER}/gcc_64/plugins/xcbglintegrations/{libqxcb-egl-integration.so,libqxcb-glx-integration.so} ${ASSEMBLY_PATH}/plugins/xcbglintegrations/

# copy styles
rsync --info=progress2 -az ${ACTIVE_SRC_PATH}/MusenGUI/styles/musen_style1.qss ${ASSEMBLY_PATH}/styles/

# create an archive
CURR_PATH=${PWD}
cd ${ACTIVE_WORK_PATH}
tar -czvf ${ACTIVE_BUILD_PATH}/${ASSEMBLY_DIR}.tar.gz ${ASSEMBLY_DIR} --remove-files
cd ${CURR_PATH}
