FROM ubuntu:18.04

RUN apt-get update && apt install -y \
build-essential \
libssl-dev \
git \
wget \
make

# cmake
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.tar.gz" -q -O - \
    | tar -xz --strip-components=1 -C /usr/local

COPY . /milvus-sdk-cpp
RUN cd /milvus-sdk-cpp && git submodule update --init

# build
RUN cd /milvus-sdk-cpp && mkdir -p build && cd build && cmake ../ && make -j 20
