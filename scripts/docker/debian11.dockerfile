FROM debian:11
LABEL description="Debian 11 to build MUSEN"
LABEL org.opencontainers.image.source="https://github.com/msolids/musen"
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /root

# Install default CUDA 11.2
RUN sed -i 's/ main$/ main contrib non-free/' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        nano rsync wget ca-certificates \
        build-essential zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev \
        nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Install CMake 3.31 (Debian 11 ships too old 3.18)
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
        | tar xz --strip-components=1 -C /usr/local

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
