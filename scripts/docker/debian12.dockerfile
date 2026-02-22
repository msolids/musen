FROM debian:12
LABEL description="Debian 12 to build MUSEN"
LABEL org.opencontainers.image.source="https://github.com/msolids/musen"
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /root

# Install default CUDA 11.8
RUN sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        nano rsync \
        build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt6opengl6-dev libglu1-mesa-dev \
        nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
