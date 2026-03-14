FROM ubuntu:20.04
LABEL description="Ubuntu 20.04 to build MUSEN"
LABEL org.opencontainers.image.source="https://github.com/msolids/musen"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /root

# Install default CUDA 10.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nano rsync \
        build-essential wget ca-certificates zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev \
        nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Install CMake 3.31 (Ubuntu 20.04 ships too old 3.16)
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
        | tar xz --strip-components=1 -C /usr/local

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
