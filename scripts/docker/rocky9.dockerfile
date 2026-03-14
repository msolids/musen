FROM rockylinux/rockylinux:9-ubi
LABEL description="Rocky Linux 9 to build MUSEN"
LABEL org.opencontainers.image.source="https://github.com/msolids/musen"
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /root

# System packages
RUN dnf install -y --setopt=install_weak_deps=False dnf-plugins-core && \
    dnf config-manager --set-enabled crb && \
    dnf install -y --setopt=install_weak_deps=False \
        nano rsync wget ca-certificates \
        gcc-c++ libstdc++-static make zlib-devel zlib-static qt5-qtbase-devel mesa-libGLU-devel && \
    dnf clean all

# Install CMake 3.31 (Rocky 9 ships too old 3.20)
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
        | tar xz --strip-components=1 -C /usr/local

# Build protobuf from source (static lib not available)
ARG PROTOBUF_VERSION=3.21.12
RUN wget -q https://github.com/protocolbuffers/protobuf/archive/refs/tags/v${PROTOBUF_VERSION}.tar.gz -O protobuf.tar.gz && \
    tar xzf protobuf.tar.gz && rm -f protobuf.tar.gz && \
    cd protobuf-${PROTOBUF_VERSION} && mkdir build && cd build && \
    cmake ../cmake -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make -j$(nproc) && make install && \
    cd /root && rm -rf protobuf-${PROTOBUF_VERSION}

# Install CUDA
ARG CUDA_VERSION=12.8
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf install -y --setopt=install_weak_deps=False \
        cuda-compiler-${CUDA_VERSION//./-} cuda-cudart-devel-${CUDA_VERSION//./-} \
        libcurand-devel-${CUDA_VERSION//./-} cuda-cccl-${CUDA_VERSION//./-} && \
    dnf clean all
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
