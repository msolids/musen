#!/bin/bash

#######################################################################################
# Installs the required version of CUDA and removes the installed ones. 
# Requires common_config.sh in the same directory to run.
# Run it as 
# $ chmod +x ./install_cuda.sh
# $ ./install_cuda.sh
# Running as 
# $ ./install_cuda.sh
# $ ./install_cuda.sh -l=host
# $ ./install_cuda.sh --location=host
# is equivalent and will perform installation on a local computer.
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

# get information about distro
DISTRO_NAME=$(lsb_release -i | cut -d$'\t' -f2 | cut -d' ' -f2 | tr '[:upper:]' '[:lower:]')
DISTRO_VERSION=$(lsb_release -r | cut -d$'\t' -f2 | tr -d .)
DISTRO=${DISTRO_NAME}${DISTRO_VERSION}
ARCH=$(uname -m)

case ${LOCATION} in
host)

	# check current version
	CURRENT_VERSION="$(/usr/local/cuda-${CUDA_VER}/bin/nvcc --version | tail -n2 | head -n1 | cut -d',' -f2 | cut -d' ' -f3)"
	if [ "$(printf '%s\n' "${CUDA_VER}" "${CURRENT_VERSION}" | sort -V | head -n1)" = "${CUDA_VER}" ]; then 
		echo "cuda " ${CURRENT_VERSION} " already installed."
		exit 1
	fi
	
	# try to uninstall previous versions
	/usr/local/cuda-10.0/bin/cuda-uninstaller
	/usr/local/cuda-10.1/bin/cuda-uninstaller
	/usr/local/cuda-10.2/bin/cuda-uninstaller
	/usr/local/cuda-11.0/bin/cuda-uninstaller
	/usr/local/cuda-11.1/bin/cuda-uninstaller
	/usr/bin/nvidia-uninstall
	apt -y --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"

	KERNEL_RELEASE=$(uname -r)
	if [[ "${KERNEL_RELEASE}" == *"WSL"* ]]; then # WSL
		# install repository meta data
		add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/ /"		
	else # Ununtu
		# install kernel headers and development packages for the currently running kernel
		apt install linux-headers-${KERNEL_RELEASE}
		# install repository meta data
		dpkg -i cuda-repo-${DISTRO}_${CUDA_VER_MAJOR}-${CUDA_VER_MINOR}_${ARCH}.deb
	fi

	# install the CUDA public GPG key
	apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/3bf863cc.pub

	# pin file to prioritize CUDA repository
	wget https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-${DISTRO}.pin
	mv cuda-${DISTRO}.pin /etc/apt/preferences.d/cuda-repository-pin-600

	# update apt repository
	apt update

	# install cuda
	apt -y install cuda-toolkit-${CUDA_VER_MAJOR}-${CUDA_VER_MINOR}

	# post-install actions
	export PATH=/usr/local/cuda-${CUDA_VER_MAJOR}.${CUDA_VER_MINOR}/bin${PATH:+:${PATH}}
	
	# if another gcc is installed with cuda, add it to alternatives list
	CURRENT_GCC_VERSION="$(gcc -dumpversion)"
	if [[ ${CURRENT_GCC_VERSION} ]]; then
		update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${CURRENT_GCC_VERSION} ${CURRENT_GCC_VERSION} --slave /usr/bin/g++ g++ /usr/bin/g++-${CURRENT_GCC_VERSION} --slave /usr/bin/gcov gcov /usr/bin/gcov-${CURRENT_GCC_VERSION}
	fi
	
	;;
	
*)
	echo "Error! Unknown location: " "${LOCATION}"
	exit 1
	;;
esac
