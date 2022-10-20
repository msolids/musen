#!/bin/bash

#######################################################################################
# Top-level script to perform compilation of MUSEN.
# Gathers and copies source files, compiles MUSEN, assembles a GUI package (if needed), 
# and copies the results back to ${MUSEN_RESULTS_PATH}.
# Requires common_config.sh, copy_source_files.sh, make_musen_centos7.sh, musen.sh in the same directory to run.
#
# To select version of MUSEN to build, run the script as
# $ ./compile_on_centos7.sh [--target=cli] [--target=gui] [--target=matlab]
# or 
# $ ./compile_on_centos7.sh [-t=c] [-t=g] [-t=m]
# where 
# cli/c    - command line version   (cmusen)
# gui/g    - GUI version            (musen)
# matlab/m - matlab library version (mmusen)
# Running the script without target parameters will build all versions. 
# Running the script with at least one target will disable the rest not mentioned targets.
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

# build arguments line with selected targets
TARGETS_ARGS=""
if [[ ${BUILD_CLI} == "yes" ]]; then TARGETS_ARGS=${TARGETS_ARGS}" --target=cli";    fi
if [[ ${BUILD_GUI} == "yes" ]]; then TARGETS_ARGS=${TARGETS_ARGS}" --target=gui";    fi
if [[ ${BUILD_MAT} == "yes" ]]; then TARGETS_ARGS=${TARGETS_ARGS}" --target=matlab"; fi
echo "targets: " ${TARGETS_ARGS}

# copy files
echo "Gather source files"
./copy_source_files.sh --location=host

# sync script files
echo "Copy scripts"
rsync --info=progress2 -avz --delete ${PWD}/musen.sh ${MUSEN_WORK_PATH}

# remove old build files
echo "Remove old build files"
rm -rf ${MUSEN_BUILD_PATH}

# compile
echo "Compile MUSEN"
./make_musen_centos7.sh ${TARGETS_ARGS}

# assembling gui version
if [[ ${BUILD_GUI} == "yes" ]]; then
	echo "Assembling GUI package"
	./assemble_gui.sh --location=host
fi

# copy compiled
echo "Copy compiled files"
mkdir -p ${MUSEN_RESULTS_PATH}
rsync --ignore-missing-args --remove-source-files --info=progress2 -rloDvz ${MUSEN_BUILD_PATH}/{musen_gui.tar.gz,cmusen,mmusen.mexa64} ${MUSEN_RESULTS_PATH}
