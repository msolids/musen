#!/bin/bash

#######################################################################################
# Copies all files required by to build MUSEN. 
# Requires common_config.sh in the same directory to run.
# Run it as 
# $ chmod +x ./copy_source_files.sh
# $ ./copy_source_files.sh
# Running as 
# $ ./copy_source_files.sh
# will copy files to the current directory.
# Running as 
# $ ./copy_source_files.sh -l=host
# $ ./copy_source_files.sh --location=host
# will copy files to the directory defined by MUSEN_SRC_PATH.
# Running as 
# $ ./copy_source_files.sh -l=hpc5
# $ ./copy_source_files.sh --location=hpc5
# will copy files to the directory defined by MUSEN_HPC5_SRC_PATH on HPC5.
#######################################################################################

# directories to search for files
DIRS=(
CMusen
Databases
Models
Modules
MusenGUI
QTDialogs
Version
)

# file extensions to copy
INCLUDE_EXTENSIONS=(
"*.cpp"
"*.h"
"*.proto"
"*.cuh"
"*.cu"
"*.ui"
"*.png"
"*.glsl"
"*.qss"
"*.qrc"
)

# directories to exclude from copy
EXCLUDE_DIRS=(
"GeneratedFiles"
"Templates"
"Visual Studio 2017"
"Visual Studio 2019"
)

# make the config file visible regardless of calling location and load it
PATH_PREFIX="${BASH_SOURCE%/*}" 
if [[ ! -d "${PATH_PREFIX}" ]]; then PATH_PREFIX="${PWD}"; fi
. "${PATH_PREFIX}/common_config.sh"

# parse arguments
LOCATION=none
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

# determine where to copy files
case ${LOCATION} in
none)
	SRC_PATH=${PWD}/${MUSEN_SRC_DIR}
	
	;;
	
host)
	SRC_PATH=${MUSEN_SRC_PATH}
	
	;;
	
hpc5)
	# local temporary copy
	SRC_PATH=/tmp/MUSEN/${MUSEN_SRC_DIR}
	rm -r ${SRC_PATH} 2>/dev/null
	mkdir -p ${SRC_PATH}
	
	;;
	
*)
	echo "Error! Unknown location: " "${LOCATION}"
	exit 1
	;;
esac

# build command arguments
INCLUDE_ARGS=''
for VAL in "${INCLUDE_EXTENSIONS[@]}"; do 
	INCLUDE_ARGS="${INCLUDE_ARGS}--include \""${VAL}"\" "
done
EXCLUDE_ARGS=''
for VAL in "${EXCLUDE_DIRS[@]}"; do 
	EXCLUDE_ARGS="${EXCLUDE_ARGS}--exclude \""${VAL}"\" "
done

# copy sources
for DIR in "${DIRS[@]}"; do 
	mkdir -p ${SRC_PATH}/${DIR}
	eval rsync --info=progress2 -az --prune-empty-dirs --delete ${EXCLUDE_ARGS} --include \"'*/'\" ${INCLUDE_ARGS} --exclude \"'*'\" \"${PWD}/../${DIR}/\" \"${SRC_PATH}/${DIR}/\"
done

# create a hash header for versioning
./generate_hash_header.sh
mv -f ./BuildHash.h ${SRC_PATH}/Version/

# copy license file
cp ${PWD}/../LICENSE ${SRC_PATH}/

# copy gathered files to a remote server if necessary
case ${LOCATION} in
hpc5)
	# move local temporary copy to the server
	rsync --info=progress2 -avz --delete -e ssh ${SRC_PATH} ${HPC5_USER}@hpc5.rz.tu-harburg.de:${MUSEN_HPC5_WORK_PATH}
	rm -r ${SRC_PATH}
	
	;;
	
*)
	;;
esac
