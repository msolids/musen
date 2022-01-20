# MUSEN - GPU-accelerated DEM simulation framework
- For more information, please check the [documentation](https://msolids.net/documentation).
- [Video introduction](https://youtu.be/bH1xydzdrGY)
- To refer MUSEN please use [Dosta et al., 2020](https://doi.org/10.1016/j.softx.2020.100618)
- [New versions and updates](https://msolids.net/musen/download)

# Requirements 
MUSEN should install and work on all latest versions of Windows or Linux (Ubuntu or Red Head).
Requires [Visual C++ Redistributable for Visual Studio 2019](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) to run on Windows.


# Compilation on Windows
A fully functional version can be compiled and built with Microsoft Visual Studio 2019. 

## Requirements on Windows
- [Microsoft Visual Studio 2019](https://visualstudio.microsoft.com/downloads/)
- [Qt 5.15.2 msvc2019_64](https://download.qt.io/archive/online_installers/4.0/)
- [Qt Visual Studio Tools for Visual Studio 2019](https://marketplace.visualstudio.com/items?itemName=TheQtCompany.QtVisualStudioTools2019)
- [CUDA 11.2](https://developer.nvidia.com/cuda-downloads)
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
2. Setup Qt Visual Studio Tools extension to point to the installed Qt libraries. In Visual Studio 2019, go to Extensions → Qt VS Tools → Qt Options → Add → ... → Navigate in the Qt installation directory to `X:/path/to/Qt/5.15.2/msvc2019_64` → OK.
3. Prepare third-party statically linked libraries: zlib, protobuf. To do this, navigate to `X:/path/to/msolids/MUSEN/ExternalLibraries/` and execute files `RunZLibCompile.bat` and `RunProtobufCompile.bat`. They will download and build all the required libraries by executing files `CompileZLib.ps1`, `CompileProtobuf.ps1`.
4. Open `X:/path/to/msolids/MUSEN/MUSEN/musen.sln` file with Visual Studio and build the solution.


# Compilation for Linux on Windows with WSL (Windows Subsystem for Linux)
A fully functional version can be compiled and built with cmake and gcc in WSL. 

## Build in WSL:
1. Enable the Windows Subsystem for Linux. Open PowerShell as Administrator and run:
```PowerShell
$ dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```
2. Update to WSL 2. Requirements: Windows Version 1903 or higher, with Build 18362 or higher. If you don't want to upgrade, restart your computer and move to step 3.
	2.1. Enable Virtual Machine feature. Open PowerShell as Administrator and run:
	```PowerShell
	$ dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
	```
	2.2. Download and install the latest [Linux kernel update package](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi).
	2.3. Set WSL 2 as your default version. Open PowerShell as Administrator and run:
	```PowerShell
	$ wsl --set-default-version 2
	```
3. Open the [Microsoft Store](https://aka.ms/wslstore) and install [Ubuntu 18.04 LTS](https://www.microsoft.com/store/apps/9N9TNGVNDL3Q) distribution.
4. Launch the installed distribution and follow the instructions for initial setup.
5. Login to your distribution and update it by running:
```sh
$ sudo apt update
$ sudo apt upgrade
```
6. Install all required tools and libraries. E.g. if the project is located at `C:/Projects/msolids/`:
```sh
$ cd /mnt/c/Projects/msolids/MUSEN_Linux
$ chmod +x ./install_prerequisites_host.sh
$ sudo ./install_prerequisites_host.sh
```
7. Further you can compile MUSEN either with Visual Studio (Step 7.a) or directly inside Ubuntu (Step 7.b)
7.a Open `C:/Projects/msolids/MUSEN/MUSEN/musen.sln` file with Visual Studio. In Solution Explorer under `Installers` folder select `LinuxBuildWSL` project, then from the main menu navigate to (Project → Properties → Configuration Properties → Linux Build Settings) and select MUSEN versions that you want to build. Run building project `LinuxBuildWSL` (Build → Build Selection).
7.b Compile MUSEN by running:
```sh
$ chmod +x ./compile_on_host.sh
$ ./compile_on_host.sh
```
8. The built executables will be placed in `C:/Projects/msolids/MUSEN_Linux/compiled/`.


# Compilation on Linux
A fully functional version can be compiled and built with cmake and gcc. 

## Requirements on Linux
- gcc-9, g++-9
- cmake 3.18.0
- qt 5.15.2 gcc_64 (Optional. Needed to build MUSEN with GUI)
- cuda 11.2
- zlib 1.2.11
- protobuf 3.14.0
- (optional) MATLAB R2019a 

## Build on Linux (Ubuntu version 18.04 or higher)
1. Navigate to `/path/to/msolids/MUSEN_Linux/`
2. Install all required build tools and third-party libraries executing file `install_prerequisites_host.sh`. Alternatively, install all required build tools (gcc, cmake, cuda) either manually or executing files `install_gcc.sh`, `install_cmake.sh`, `install_cuda`, then compile all third-party libraries executing files `install_zlib.sh`, `install_protobuf.sh`, `install_qt.sh`.
3. Start compilation by executing file `compile_on_host.sh`.
4. Built executables can be found in `/path/to/msolids/MUSEN_Linux/compiled`.

## Run a GUI version on Linux
1. Install required additional libraries
```sh
$ sudo apt install libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0
```
2. Extract the compiled GUI version, e.g.
```sh
$ tar xfvz /path/to/msolids/MUSEN_Linux/compiled/vX.X.X_name/musen_gui.tar.gz
```
3. Run the startup script
```sh
$ chmod +x musen_gui/musen.sh
$ musen_gui/musen.sh
```

# Code organization
- Databases - agglomerates, geometries and materials databases
- ExternalLibraries - external libraries used in MUSEN (zlib and protobuf)
- GeneralFunctions - main functions and types used in MUSEN 
- GeneralFunctions\SimResultsStorage - low-level functions for data handling (load and save data)
- MUSEN
- MUSEN\CMusen - command-line version of MUSEN
- MUSEN\Models - contact models (particle-particle, particle-wall, solid bonds, ...)
- MUSEN\Modules 
- MUSEN\Modules\BondsGenerator - generate bonds between particles
- MUSEN\Modules\ContactAnalyzer - general functions for fast contacts detection 
- MUSEN\Modules\FileManager - set of functions to convert, merge or modify .mdem files with simulation results
- MUSEN\Modules\Geometries - set of classes and functions to work with geometrical objects
- MUSEN\Modules\ObjectsGenerator - dynamically generates particles or agglomerates during simulation
- MUSEN\Modules\PackageGenerator - generate packing of particles prior simulation
- MUSEN\Modules\ResultsAnalyzer - analyzer of simulation results (export necessary data to csv files)
- MUSEN\Modules\ScriptInterface - analyze input scripts for command-line version
- MUSEN\Modules\SimplifiedScene - simplified entity generated from SystemStructure and which is used during simulation
- MUSEN\Modules\Simulator - CPU and GPU simulators
- MUSEN\Modules\SystemStructure - main entity which stores the information about whole scene
- MUSEN\QTDialog - Qt dialog for main window
- MUSEN_Linux - set of scripts to compile MUSEN for linux
- QTDialogs - different Qt-based dialogs of GUI


# Third-party tools and libraries
- [CUDA 11.2](https://developer.nvidia.com/cuda-zone) – Nvidia Corporation - [NVIDIA License](https://docs.nvidia.com/cuda/pdf/EULA.pdf)
- [Inno Setup 6.1.2](https://jrsoftware.org/isinfo.php) - Jordan Russell - [Modified BSD License](http://www.jrsoftware.org/files/is/license.txt)
- [Protobuf 3.14.0](https://developers.google.com/protocol-buffers/) – Google Inc. - [BSD License](https://github.com/protocolbuffers/protobuf/blob/master/LICENSE)
- [Qt 5.15.2](https://www.qt.io/) – The Qt Company - [LGPLv3 License](https://doc.qt.io/qt-5/lgpl.html)
- [Visual Studio Community 2019](https://visualstudio.microsoft.com/vs/) - Microsoft Corporation - [Microsoft Software License Terms](https://visualstudio.microsoft.com/license-terms/mlt031819/)
- [zlib v1.2.11](https://www.zlib.net/) – Jean-loup Gailly and Mark Adler - [zlib License](https://www.zlib.net/zlib_license.html)
