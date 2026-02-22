FROM debian:13
LABEL description="Debian 13 to build MUSEN"
LABEL org.opencontainers.image.source="https://github.com/msolids/musen"
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /root

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nano rsync wget ca-certificates linux-headers-generic \
        build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt6opengl6-dev libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA from NVIDIA repo (default CUDA 12.4 doesn't support default GCC 14)
ARG CUDA_VERSION=13.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION//./-} cuda-cudart-dev-${CUDA_VERSION//./-} \
        libcurand-dev-${CUDA_VERSION//./-} cuda-cccl-${CUDA_VERSION//./-} && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
