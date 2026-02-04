FROM --platform=linux/amd64 ubuntu:latest

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive
# Install Python and pip
RUN apt-get update \
&& apt-get install -y --no-install-recommends python3 \
python3-pip \
build-essential \
git \
libeigen3-dev \
python3-dev \
cmake \
ninja-build \
llvm \
pkg-config \
libcgal-dev \
nlohmann-json3-dev \
libfmt-dev \
lsb-release \
curl \
&& rm -rf /var/lib/apt/lists/*

RUN python3 -m pip config set global.break-system-packages true
RUN pip install viser numpy yourdfpy trimesh robot-descriptions