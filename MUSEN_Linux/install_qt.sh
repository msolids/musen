#!/bin/bash

#######################################################################################
# Installs the required version of Qt. qt_installer_script.qt is required to run.
# Requires common_config.sh in the same directory to run.
# Run it as 
# $ chmod +x ./install_qt.sh
# $ ./install_qt.sh
# Running as 
# $ ./install_qt.sh
# $ ./install_qt.sh -l=host
# $ ./install_qt.sh --location=host
# is equivalent and will perform installation on a local computer.
# Run as 
# $ ./install_qt.sh -l=hpc5
# $ ./install_qt.sh --location=hpc5
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
	
	# installation path
	QT_INSTALL_PATH=${MUSEN_EXTERNAL_LIBS_PATH}/qt

	# current path
	RUN_DIR=${PWD}
	
	# if working directory does not exist yet
	if [ ! -d "${MUSEN_WORK_PATH}" ]; then
		# create working directory
		mkdir -p ${MUSEN_WORK_PATH}
		# set proper permissions for current user
		chown ${SUDO_USER}.${SUDO_USER} ${MUSEN_WORK_PATH}
	fi
	# enter build directory
	cd ${MUSEN_WORK_PATH}

	# install additional libraries
	apt -y install libxkbcommon-x11-0
	apt -y install libwayland-cursor0
	apt -y install libglu1-mesa-dev
	
	;;
	
hpc5)
	# installation path
	QT_INSTALL_PATH=${PWD}/${MUSEN_EXTERNAL_LIBS_DIR}/qt

	# current path
	RUN_DIR=${PWD}

	;;
	
*)
	echo "Error! Unknown location: " "${LOCATION}"
	exit 1
	;;
esac

# check current version
CURRENT_VERSION="$(${QT_INSTALL_PATH}/${QT_VER}/gcc_64/bin/qmake --version | tail -n1 | cut -d' ' -f4)"
if [ "$(printf '%s\n' "${QT_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${QT_VER}" ]; then 
	echo "qt " ${CURRENT_VERSION} " already installed."
	exit 1
fi

# file names
QT_INSTALLER_FILE=qt-unified-linux-x64-${QT_INSTALLER_VER}-online.run

# clear old
rm -rf ${QT_INSTALL_PATH}

# download installer
wget https://download.qt.io/archive/online_installers/${QT_INSTALLER_VER_MAJOR}.${QT_INSTALLER_VER_MINOR}/${QT_INSTALLER_FILE}

# install qt
chmod +x ${QT_INSTALLER_FILE}
./${QT_INSTALLER_FILE} install qt.qt${QT_VER_MAJOR}.${QT_VER_MAJOR}${QT_VER_MINOR}${QT_VER_BUILD}.gcc_64 --root ${QT_INSTALL_PATH} --email musen.public@gmail.com --password VeryLong+SecurePasswordForMUSENDevelopers --accept-licenses --accept-obligations --auto-answer telemetry-question=No,AssociateCommonFiletypes=No --platform minimal --no-default-installations --no-force-installations --confirm-command

# remove unnecessary 
rm -rf ${QT_INSTALL_PATH}/dist
rm -rf ${QT_INSTALL_PATH}/Docs
rm -rf ${QT_INSTALL_PATH}/Examples
rm -rf ${QT_INSTALL_PATH}/installerResources
rm -rf ${QT_INSTALL_PATH}/Licenses
rm -rf ${QT_INSTALL_PATH}/Tools
rm -rf ${QT_INSTALL_PATH}/components.xml
rm -rf ${QT_INSTALL_PATH}/InstallationLog.txt
rm -rf ${QT_INSTALL_PATH}/installer.dat
rm -rf ${QT_INSTALL_PATH}/MaintenanceTool
rm -rf ${QT_INSTALL_PATH}/MaintenanceTool.dat
rm -rf ${QT_INSTALL_PATH}/MaintenanceTool.ini
rm -rf ${QT_INSTALL_PATH}/network.xml
rm -rf ${QT_INSTALL_PATH}/QtIcon.png
rm -rf ${QT_INSTALL_PATH}/${QT_VER}/sha1s.txt
rm -rf ${QT_INSTALL_PATH}/${QT_VER}/gcc_64/doc

# clean
rm ${QT_INSTALLER_FILE}
