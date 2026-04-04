# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
# See LICENSE file for license and warranty information.

FROM ubuntu:24.04
LABEL description="Ubuntu 24.04 to build MUSEN"
LABEL org.opencontainers.image.source="https://github.com/msolids/musen"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /root

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget ca-certificates \
        build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt6opengl6-dev libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA from NVIDIA repo (default is CUDA 12.0)
ARG CUDA_VERSION=12.6
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION//./-} libcurand-dev-${CUDA_VERSION//./-} cuda-cccl-${CUDA_VERSION//./-} && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
