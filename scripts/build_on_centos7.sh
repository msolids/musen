# install git and download MUSEN source
sudo yum install git
git clone --branch gpu_dynamic_gen --depth 1 https://github.com/msolids/musen.git

# install suitable gcc/g++ version and load it
sudo yum install devtoolset-7-gcc*
echo "source scl_source enable devtoolset-7" >> ~/.bashrc
source ~/.bashrc

# install other build tools and MUSEN dependencies 
sudo yum install make cmake zlib-devel qt5-qtbase-devel

# OpenGL driver
#sudo yum install mesa-dri-drivers

# install CUDA 10.2
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
sudo yum clean all
sudo yum -y install cuda-10-2
echo 'export PATH=/usr/local/cuda-10.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# build a newer version of protobuf
mkdir musen_ext_libs
cd musen_ext_libs/
../musen/scripts/build_protobuf.sh

# make an internal script executable
cd ../musen/
chmod +x ./Version/generate_build_version.sh

# build MUSEN with custom protobuf and a newer OpenGL library
mkdir install
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_PREFIX_PATH=~/musen_ext_libs/protobuf/ -DOpenGL_GL_PREFERENCE=GLVND
cmake --build . --parallel $(nproc)
make install