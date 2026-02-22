FROM ubuntu:22.04
LABEL description="Ubuntu 22.04 to build MUSEN"
LABEL org.opencontainers.image.source="https://github.com/msolids/musen"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /root

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nano rsync wget ca-certificates linux-headers-generic \
        build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA from NVIDIA repo (default CUDA 11.5 doesn't support default GCC 11)
ARG CUDA_VERSION=11.7
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-compiler-${CUDA_VERSION//./-} cuda-cudart-dev-${CUDA_VERSION//./-} libcurand-dev-${CUDA_VERSION//./-} cuda-cccl-${CUDA_VERSION//./-} && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
