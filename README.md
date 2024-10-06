# MUSEN - GPU-accelerated DEM simulation framework
- For more information, please check the [documentation](https://msolids.net/documentation). 
- [Video introduction](https://youtu.be/bH1xydzdrGY)
- To refer MUSEN please use [Dosta et al., 2020](https://doi.org/10.1016/j.softx.2020.100618).
- [New versions and updates](https://github.com/msolids/musen/releases).


# Requirements 
MUSEN should install and work on all latest versions of Windows or Linux (Ubuntu or Red Head).
Requires [Visual C++ Redistributable for Visual Studio 2022](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version) to run on Windows.


# Compilation on Windows
A fully functional version can be compiled and built with Microsoft Visual Studio 2022. 

## Requirements on Windows
- [Microsoft Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
- [Qt 5.15.2 msvc2019_64](https://download.qt.io/archive/online_installers/4.0/)
- [Qt Visual Studio Tools for Visual Studio 2022](https://marketplace.visualstudio.com/items?itemName=TheQtCompany.QtVisualStudioTools2022)
- [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64)
- [Git](https://git-scm.com/downloads)
- [CMake](https://cmake.org/download/)
- (optional) MATLAB R2019a

## Build on Windows
1. Download and install all [requirements](#requirements-on-windows) on your computer.
	
	1.1. Microsoft Visual Studio: select "Desktop Development with C++" when choosing what components to install.
	
	1.2. Qt: You will need to create a free Qt-account to run installation. When selecting components, choose Qt → Qt 5.15.2 → MSVC 2019 64-bit. 
	
	1.3. CUDA: Download the specified version of CUDA and install the configuration proposed by the setup utility.
	
	1.4. Use the last available version of CMake and select the option "Add to system path" if available during installation.
	
	1.5. Use the last available version of Git.
2. Setup Qt Visual Studio Tools extension to point to the installed Qt libraries. In Visual Studio, go to Extensions → Qt VS Tools → Qt Options → Add → ... → Navigate in the Qt installation directory to `X:/path/to/Qt/5.15.2/msvc2019_64` → OK.
3. Prepare third-party statically linked libraries: zlib, protobuf. To do this, navigate to `X:/path/to/msolids/MUSEN/ExternalLibraries/` and execute files `RunZLibCompile.bat` and `RunProtobufCompile.bat`. They will download and build all the required libraries by executing files `CompileZLib.ps1`, `CompileProtobuf.ps1`.
4. Open `X:/path/to/msolids/MUSEN/MUSEN/musen.sln` file with Visual Studio and build the solution.


# Compilation for Linux on Windows with WSL (Windows Subsystem for Linux)
A fully functional version can be compiled and built in WSL. 

## Build in WSL:
1. Enable the Windows Subsystem for Linux. Open PowerShell and run:
```PowerShell
wsl --install
```
Additional information [here](https://learn.microsoft.com/en-us/windows/wsl/install).

2. Install Ubuntu
```PowerShell
wsl --install -d Ubuntu-22.04
```

3. Launch the installed distribution and follow the instructions for initial setup.

4. Login to your distribution and update it by running:
```sh
sudo apt update
sudo apt upgrade
```

5. Install all required tools and libraries, as described in [Build on Linux](#build-on-linux). 

6. Compile MUSEN either with Visual Studio (step 6.a) or directly in Ubuntu (step 6.b)
	
	6.a Open `.../musen/musen.sln` file with Visual Studio. In Solution Explorer under `Installers` folder select `LinuxBuildWSL` project, then from the main menu navigate to (Project → Properties → Configuration Properties → Linux Build Settings) and select MUSEN versions that you want to build. Run building project `LinuxBuildWSL` (Build → Build Selection).
	
	6.b Compile MUSEN as described in [Build on Linux](#build-on-linux).  
	
7. The built executables will be placed in `...musen/Installers/Installers/`.


# Compilation on Linux
A fully functional version can be compiled and built with cmake and gcc. 

## Minimum requirements on Linux
- gcc-7.5, g++-7.5
- cmake 3.0.0
- protobuf 3.0.0
- qt 5.9.5
- cuda 9.1
The versions of CUDA and C++ compiler must be compatible. See compatibility list e.g. [here](https://gist.github.com/ax3l/9489132#nvcc).

## Build on Linux 
Tested on Ubuntu 18.04, 20.04, 22.04.
1. Change the current working directory to the desired location and download the MUSEN code:
```sh
cd /path/to/desired/location/
git clone --depth 1 https://github.com/msolids/musen.git
cd musen
```
2. Install required tools and libraries.
```sh
sudo apt install build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev
```
3. Install CUDA
```sh
sudo apt install nvidia-cuda-toolkit
```
or in case of compatibility issues (usually on Ubuntu 22.04), using official [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) or running the script for Ubuntu:
```sh
./scripts/install_cuda.sh
exec bash
```
4. Build MUSEN
```sh
mkdir install
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --parallel $(nproc)
make install
```
5. Built executables can be found in 
```sh
cd ../install
```

# Code organization
- CMusen - command-line version of MUSEN
- Databases - agglomerates, geometries and materials databases
- Documentation - manuals
- ExternalLibraries - external libraries used in MUSEN on Windows (zlib and protobuf)
- GeneralFunctions - main functions and types used in MUSEN 
- Installers - scripts and data needed to build installers on Windows
- Models - contact models (particle-particle, particle-wall, solid bonds, etc.)
- Modules\BondsGenerator - generate bonds between particles
- Modules\ContactAnalyzer - functions for contacts detection 
- Modules\FileManager - set of functions to convert, merge or modify .mdem files with simulation results
- Modules\GeneralSources - general components
- Modules\Geometries - set of classes and functions to work with geometrical objects
- Modules\ObjectsGenerator - dynamic particles or agglomerates generator
- Modules\PackageGenerator - generate packing of particles prior simulation
- Modules\ResultsAnalyzer - analyzer of simulation results (export necessary data to csv files)
- Modules\ScriptInterface - analyze input scripts for command-line version
- Modules\SimplifiedScene - simplified entity generated from SystemStructure and which is used during simulation
- Modules\SimResultsStorage - low-level functions for data handling (load and save data)
- Modules\Simulator - CPU and GPU simulators
- Modules\SystemStructure - main entity which stores the information about whole scene
- MusenGUI - graphical version of MUSEN
- QTDialog - Qt-based dialogs for graphical user interface
- Version - version information


# Third-party tools and libraries
- [CUDA 11.8](https://developer.nvidia.com/cuda-zone) – Nvidia Corporation – [NVIDIA License](https://docs.nvidia.com/cuda/pdf/EULA.pdf)
- [Inno Setup 6.1.2](https://jrsoftware.org/isinfo.php) – Jordan Russell – [Modified BSD License](http://www.jrsoftware.org/files/is/license.txt)
- [Protobuf 3.21.12](https://developers.google.com/protocol-buffers/) – Google Inc. – [BSD License](https://github.com/protocolbuffers/protobuf/blob/master/LICENSE)
- [Qt 5.15.2](https://www.qt.io/) – The Qt Company – [LGPLv3 License](https://doc.qt.io/qt-5/lgpl.html)
- [Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/) – Microsoft Corporation – [Microsoft Software License Terms](https://visualstudio.microsoft.com/license-terms/mlt031819/)
- [zlib v1.3.1](https://www.zlib.net/) – Jean-loup Gailly and Mark Adler – [zlib License](https://www.zlib.net/zlib_license.html)
