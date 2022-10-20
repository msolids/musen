#!/bin/bash

#######################################################################################
# Installs the required version of Qt. qt_installer_script.qt is required to run.
# Requires common_config.sh in the same directory to run.
# Run it as 
# $ chmod +x ./install_qt_centos7.sh
# $ sudo ./install_qt_centos7.sh
#######################################################################################

# make the config file visible regardless of calling location and load it
PATH_PREFIX="${BASH_SOURCE%/*}" 
if [[ ! -d "${PATH_PREFIX}" ]]; then PATH_PREFIX="${PWD}"; fi
. "${PATH_PREFIX}/common_config.sh"

# check for running as root
if [[ ${EUID} -ne 0 ]]; then
   echo "Please run this script as root or using sudo!" 
   exit 1
fi
	
# installation path
QT_INSTALL_PATH=${MUSEN_EXTERNAL_LIBS_PATH}/qt

# check current version
CURRENT_VERSION="$(${QT_INSTALL_PATH}/${QT_VER}/gcc_64/bin/qmake --version | tail -n1 | cut -d' ' -f4)"
if [ "$(printf '%s\n' "${QT_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${QT_VER}" ]; then 
	echo "qt " ${CURRENT_VERSION} " already installed."
	exit 1
fi

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

# install required libraries
yum -y install wget
yum -y install libXcomposite
yum -y install libXext
yum -y install libXrender
yum -y install libxkbcommon-x11
yum -y install fontconfig
yum -y install libwayland-cursor
yum -y install libGL
yum -y install mesa-libGL
yum -y install mesa-libGL-devel
yum -y install mesa-libGLU-devel
yum -y install libXi-devel
yum -y install libXmu-devel
yum -y install freeglut-devel.x86_64

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
