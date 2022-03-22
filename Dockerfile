FROM ubuntu:18.04

COPY . /milvus-sdk-cpp
RUN ck /milvus-sdk-cpp && git submodule update --init

RUN apt-get update && apt install libssl-dev

# cmake
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.tar.gz" -q -O - \
    | tar -xz --strip-components=1 -C /usr/local

# build
RUN mkdir build && cd build && cmake ../ && make

# run example
RUN ./examples/simple/sdk_simple
