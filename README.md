[![Release](https://img.shields.io/github/v/release/msolids/musen)](https://github.com/msolids/musen/releases)
[![License](https://img.shields.io/badge/license-BSD_3--Clause-blue)](LICENSE)
[![Windows](https://img.shields.io/github/actions/workflow/status/msolids/musen/build_windows.yml?branch=master&label=Windows)](https://github.com/msolids/musen/actions/workflows/build_windows.yml)
[![Linux](https://img.shields.io/github/actions/workflow/status/msolids/musen/build_linux.yml?branch=master&label=Linux)](https://github.com/msolids/musen/actions/workflows/build_linux.yml)

# MUSEN

MUSEN is an open-source GPU-accelerated Discrete Element Method (DEM) simulation framework with a full-featured graphical user interface and an extensible contact model architecture, working on Windows and Linux.

**Key features:**

- GPU-accelerated simulations via CUDA
- Full-featured graphical user interface
- Bonded Particle Model
- Movable physical geometries
- Periodic boundary conditions
- Extensible contact model architecture (custom models)

## Contents

- [Links](#links)
- [Building from Source](#building-from-source)
  - [Minimum Requirements](#minimum-requirements)
  - [CMake Options](#cmake-options)
  - [Windows](#windows)
  - [Linux](#linux)
  - [Docker](#docker)
- [ModelsCreator](#modelscreator-windows)
- [Third-Party Libraries](#third-party-libraries)
- [License](#license)

## Links

- **Website:** <https://msolids.net>
- **Documentation:** <https://github.com/msolids/musen/wiki>
- **Releases:** <https://github.com/msolids/musen/releases>
- **YouTube:** <https://www.youtube.com/@musensimulations6940>
- **Video introduction:** <https://youtu.be/bH1xydzdrGY>
- **Citation:** [Dosta et al., SoftwareX, 2020](https://doi.org/10.1016/j.softx.2020.100618)

## Building from Source

MUSEN uses CMake and requires a C++17 compiler and the CUDA Toolkit. Building the GUI additionally requires Qt.

The versions of CUDA and C++ compiler must be compatible. See e.g. [CUDA-compiler compatibility list](https://gist.github.com/ax3l/9489132).

Other third-party libraries are fetched and built automatically during configuration if not found on the system.

### Minimum Requirements

| Requirement | Version |
| --- | --- |
| CMake | ≥ 3.23 |
| C++ standard | C++17 |
| CUDA Toolkit | ≥ 11 |
| Qt | 5 or 6 |

### CMake Options

| Option | Default | Description |
| --- | --- | --- |
| `MUSEN_BUILD_GUI` | `ON` | Build the Qt-based GUI application (MUSEN). Requires Qt. |
| `MUSEN_BUILD_CLI` | `ON` | Build the command-line application (CMUSEN). |
| `MUSEN_INSTALL_DATA` | `ON` | Install documentation, examples, and material databases alongside binaries. |
| `MUSEN_BUILD_INSTALLER` | `OFF` | Build the Windows installer target using Inno Setup. Windows only. |
| `MUSEN_INSTALLER_SKIP_PREBUILD` | `OFF` | Skip automatic build of binaries inside the installer target. Windows only. |

### Windows

#### Prerequisites

- Visual Studio [2019](https://aka.ms/vs/16/release/vs_community.exe), [2022](https://aka.ms/vs/17/release/vs_community.exe), or [2026](https://aka.ms/vs/18/Stable/vs_community.exe) with the "Desktop Development with C++" workload
- [CUDA Toolkit 11+](https://developer.nvidia.com/cuda-downloads) — see the compatibility table below
- [CMake 3.23+](https://cmake.org/download/)
- [Git](https://git-scm.com/install/windows)
- [Qt 6 msvc2022_64](https://download.qt.io/official_releases/online_installers/) — only required if building the GUI

#### CUDA — Visual Studio Compatibility

| CUDA | VS 2019 | VS 2022 | VS 2026 |
| --- | :---: | :---: | :---: |
| 11.0 – 11.5 | Yes | — | — |
| 11.6 – 13.1 | Yes | Yes | — |
| 13.2+ | Yes | Yes | Yes |

See the official [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) for details.

#### Build with Visual Studio

For Visual Studio 2026:

```batch
cmake -B build -G "Visual Studio 18 2026"
cmake --build build --config Release --parallel
```

For Visual Studio 2022:

```batch
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release --parallel
```

To build only the CLI (no Qt required), add `-DMUSEN_BUILD_GUI=OFF`.

Built binaries are placed in `build/Release/` by default.

#### Build with Ninja

The [Ninja](https://ninja-build.org/) generator can be used as an alternative to the Visual Studio generator. Ninja can work with CUDA and MSVC combinations that the Visual Studio generator does not officially support.

```batch
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

#### Installer

To create a Windows Installer, configure with `-DMUSEN_BUILD_INSTALLER=ON`.
Requires [Inno Setup](https://jrsoftware.org/isinfo.php) available on `PATH`.

```batch
cmake -B build -G "Visual Studio 17 2022" -DMUSEN_BUILD_INSTALLER=ON
cmake --build build --target installer --config Release
```

### Linux

Tested on Ubuntu 18/20/22/24, Debian 11/12/13, Fedora 41/42, and Rocky 8/9/10.

Install the prerequisites for your distribution, then proceed to [build](#linux-build).

#### Ubuntu

<details>
<summary>Ubuntu 18.04</summary>

Install system packages:

```sh
sudo apt-get update
sudo apt-get install -y wget ca-certificates build-essential zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev
```

Install CMake 3.23+ (system version is too old):

```sh
wget -q https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
sudo tar -xzf cmake-3.31.6-linux-x86_64.tar.gz -C /usr/local --strip-components=1
rm cmake-3.31.6-linux-x86_64.tar.gz
```

Install CUDA:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-11-0 libcurand-dev-11-0
rm cuda-keyring_1.1-1_all.deb
export PATH=/usr/local/cuda-11.0/bin:$PATH
```

</details>

<details>
<summary>Ubuntu 20.04</summary>

Install system packages:

```sh
sudo apt-get update
sudo apt-get install -y wget ca-certificates build-essential zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev
```

Install CMake 3.23+ (system version is too old):

```sh
wget -q https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
sudo tar -xzf cmake-3.31.6-linux-x86_64.tar.gz -C /usr/local --strip-components=1
rm cmake-3.31.6-linux-x86_64.tar.gz
```

Install CUDA:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-11-0 libcurand-dev-11-0
rm cuda-keyring_1.1-1_all.deb
export PATH=/usr/local/cuda-11.0/bin:$PATH
```

</details>

<details>
<summary>Ubuntu 22.04</summary>

Install system packages:

```sh
sudo apt-get update
sudo apt-get install -y wget ca-certificates linux-headers-generic build-essential zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev
```

Install CMake 3.23+ (system version is too old):

```sh
wget -q https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
sudo tar -xzf cmake-3.31.6-linux-x86_64.tar.gz -C /usr/local --strip-components=1
rm cmake-3.31.6-linux-x86_64.tar.gz
```

Install CUDA:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-11-7 libcurand-dev-11-7 cuda-cccl-11-7
rm cuda-keyring_1.1-1_all.deb
export PATH=/usr/local/cuda-11.7/bin:$PATH
```

</details>

<details>
<summary>Ubuntu 24.04</summary>

Install system packages:

```sh
sudo apt-get update
sudo apt-get install -y wget ca-certificates build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt6opengl6-dev libglu1-mesa-dev
```

Install CUDA:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-12-6 libcurand-dev-12-6 cuda-cccl-12-6
rm cuda-keyring_1.1-1_all.deb
export PATH=/usr/local/cuda-12.6/bin:$PATH
```

</details>

#### Debian

<details>
<summary>Debian 11</summary>

Install system packages:

```sh
sudo apt-get update
sudo apt-get install -y wget ca-certificates build-essential zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev
```

Install CMake 3.23+ (system version is too old):

```sh
wget -q https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
sudo tar -xzf cmake-3.31.6-linux-x86_64.tar.gz -C /usr/local --strip-components=1
rm cmake-3.31.6-linux-x86_64.tar.gz
```

Install CUDA:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-11-5 libcurand-dev-11-5 cuda-cccl-11-5
rm cuda-keyring_1.1-1_all.deb
export PATH=/usr/local/cuda-11.5/bin:$PATH
```

</details>

<details>
<summary>Debian 12</summary>

Install system packages:

```sh
sudo apt-get update
sudo apt-get install -y wget ca-certificates build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt6opengl6-dev libglu1-mesa-dev
```

Install CUDA:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-12-3 libcurand-dev-12-3 cuda-cccl-12-3
rm cuda-keyring_1.1-1_all.deb
export PATH=/usr/local/cuda-12.3/bin:$PATH
```

</details>

<details>
<summary>Debian 13</summary>

Install system packages:

```sh
sudo apt-get update
sudo apt-get install -y wget ca-certificates linux-headers-generic build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt6opengl6-dev libglu1-mesa-dev
```

Install CUDA:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-13-1 libcurand-dev-13-1 cuda-cccl-13-1
rm cuda-keyring_1.1-1_all.deb
export PATH=/usr/local/cuda-13.1/bin:$PATH
```

</details>

#### Fedora

<details>
<summary>Fedora 41</summary>

Install system packages:

```sh
sudo dnf install -y wget ca-certificates gcc-c++ libstdc++-static cmake make zlib-devel zlib-static qt6-qtbase-devel mesa-libGLU-devel
```

Install CUDA:

```sh
sudo dnf install -y 'dnf-command(config-manager)'
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo
sudo dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 libcurand-devel-12-9 cuda-cccl-12-9
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

</details>

<details>
<summary>Fedora 42</summary>

Install system packages:

```sh
sudo dnf install -y wget ca-certificates gcc-c++ libstdc++-static cmake make zlib-devel zlib-static qt6-qtbase-devel mesa-libGLU-devel
```

Install CUDA:

```sh
sudo dnf install -y 'dnf-command(config-manager)'
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo
sudo dnf install -y cuda-nvcc-13-1 cuda-cudart-devel-13-1 libcurand-devel-13-1 cuda-cccl-13-1
export PATH=/usr/local/cuda-13.1/bin:$PATH
```

</details>

#### Rocky Linux

<details>
<summary>Rocky Linux 8</summary>

Enable the PowerTools repository and install system packages:

```sh
sudo dnf config-manager --set-enabled powertools
sudo dnf install -y wget ca-certificates gcc-c++ libstdc++-static make zlib-devel zlib-static qt5-qtbase-devel mesa-libGLU-devel
```

Install CMake 3.23+ (system version is too old):

```sh
wget -q https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
sudo tar -xzf cmake-3.31.6-linux-x86_64.tar.gz -C /usr/local --strip-components=1
rm cmake-3.31.6-linux-x86_64.tar.gz
```

Install CUDA:

```sh
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 libcurand-devel-12-9 cuda-cccl-12-9
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

</details>

<details>
<summary>Rocky Linux 9</summary>

Enable the CRB repository and install system packages:

```sh
sudo dnf config-manager --set-enabled crb
sudo dnf install -y wget ca-certificates gcc-c++ libstdc++-static make zlib-devel zlib-static qt5-qtbase-devel mesa-libGLU-devel
```

Install CMake 3.23+ (system version is too old):

```sh
wget -q https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
sudo tar -xzf cmake-3.31.6-linux-x86_64.tar.gz -C /usr/local --strip-components=1
rm cmake-3.31.6-linux-x86_64.tar.gz
```

Install CUDA:

```sh
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 libcurand-devel-12-9 cuda-cccl-12-9
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

</details>

<details>
<summary>Rocky Linux 10</summary>

Enable the CRB repository and install system packages:

```sh
sudo dnf config-manager --set-enabled crb
sudo dnf install -y wget ca-certificates gcc-c++ libstdc++-static cmake make zlib-devel zlib-static qt6-qtbase-devel mesa-libGLU-devel
```

Install CUDA:

```sh
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/cuda-rhel10.repo
sudo dnf install -y cuda-nvcc-13-2 cuda-cudart-devel-13-2 libcurand-devel-13-2 cuda-cccl-13-2
export PATH=/usr/local/cuda-13.2/bin:$PATH
```

</details>

#### Linux Build

After installing the prerequisites for your distribution:

```sh
git clone https://github.com/msolids/musen.git
cd musen
cmake -B build -DCMAKE_INSTALL_PREFIX=install
cmake --build build --parallel $(nproc)
cmake --install build
```

Built executables are in `install/bin/`.

To build only the CLI (no Qt required), add `-DMUSEN_BUILD_GUI=OFF`.

### Docker

Docker build scripts are provided for all 12 tested distributions in `scripts/docker/`.

Build a Docker image and compile MUSEN inside a container. Run from the repository root:

**Linux host:**

```sh
mkdir -p output
docker build -f scripts/docker/ubuntu24.dockerfile -t musen_ubuntu24 scripts/docker
docker run --rm -v "$(pwd):/mnt/musen_src:ro" -v "$(pwd)/output:/mnt/output" musen_ubuntu24 bash -c "./build_musen.sh && cp -r ~/install/* /mnt/output/"
```

The executables will be in `output/bin/`.

**Windows host (Docker Desktop):**

Volume mounts are very slow on Docker Desktop. Copy the source into the container's native filesystem before building:

CMD:

```batch
mkdir output
docker build -f scripts/docker/ubuntu24.dockerfile -t musen_ubuntu24 scripts/docker
docker run --rm -v "%cd%:/mnt/src_host:ro" -v "%cd%/output:/mnt/output" musen_ubuntu24 bash -c "/mnt/src_host/scripts/copy_src.sh && ./build_musen.sh && cp -r ~/install/* /mnt/output/"
```

PowerShell:

```powershell
mkdir output
docker build -f scripts/docker/ubuntu24.dockerfile -t musen_ubuntu24 scripts/docker
docker run --rm -v "${PWD}:/mnt/src_host:ro" -v "${PWD}/output:/mnt/output" musen_ubuntu24 bash -c "/mnt/src_host/scripts/copy_src.sh && ./build_musen.sh && cp -r ~/install/* /mnt/output/"
```

## ModelsCreator (Windows)

ModelsCreator is a standalone CMake project for building custom DEM contact model DLLs.
It supports four model types: particle-particle, particle-wall, solid bonds, and external forces.
The resulting `.dll` files can be loaded at runtime through MUSEN's Models Manager.

ModelsCreator includes template models as starting points and example models as reference implementations.

### Location

ModelsCreator is installed alongside MUSEN at `C:\Program Files\MUSEN\ModelsCreator` by the Windows installer.

### ModelsCreator Prerequisites

- Visual Studio [2019](https://aka.ms/vs/16/release/vs_community.exe), [2022](https://aka.ms/vs/17/release/vs_community.exe), or [2026](https://aka.ms/vs/18/Stable/vs_community.exe) with the "Desktop Development with C++" workload
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [CMake 3.23+](https://cmake.org/download/)

### ModelsCreator Build

Run `OpenInVisualStudio.bat` to generate and open the Visual Studio solution, or use CMake directly, e.g.:

```batch
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

Add the directory containing the resulting `.dll` files to MUSEN's Models Manager.

## Third-Party Libraries

| Library | License |
| --- | --- |
| [CUDA](https://developer.nvidia.com/cuda-zone) | [NVIDIA License](https://docs.nvidia.com/cuda/pdf/EULA.pdf) |
| [Inno Setup](https://jrsoftware.org/isinfo.php) | [Modified BSD License](http://www.jrsoftware.org/files/is/license.txt) |
| [protobuf](https://protobuf.dev/) | [BSD License](https://github.com/protocolbuffers/protobuf/blob/master/LICENSE) |
| [Qt](https://www.qt.io/) | [LGPLv3](https://doc.qt.io/qt-6/lgpl.html) |
| [zlib](https://www.zlib.net/) | [zlib License](https://www.zlib.net/zlib_license.html) |

## License

MUSEN is licensed under the [BSD 3-Clause License](LICENSE).
