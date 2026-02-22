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
        nano rsync \
        build-essential cmake zlib1g-dev libprotobuf-dev protobuf-compiler libqt5opengl5-dev \
        nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/*

COPY --chmod=755 build_musen.sh ./build_musen.sh

CMD ["/bin/bash"]
