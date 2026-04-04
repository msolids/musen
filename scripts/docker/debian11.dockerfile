# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
# See LICENSE file for license and warranty information.

FROM debian:11
LABEL description="Debian 11 to build MUSEN"
LABEL org.opencontainers.image.source="https://github.com/msolids/musen"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /root

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget ca-certificates \
        build-essential zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CMake 3.31 (Debian 11 ships too old 3.18)
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
        | tar xz --strip-components=1 -C /usr/local

# Install CUDA from NVIDIA repo (default is CUDA 11.2)
ARG CUDA_VERSION=11.5
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION//./-} libcurand-dev-${CUDA_VERSION//./-} cuda-cccl-${CUDA_VERSION//./-} && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
