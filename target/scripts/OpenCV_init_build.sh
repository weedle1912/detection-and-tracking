#!/bin/bash

# ***
# Download source and create Makefile for OpenCV
# - IN_DIR: Install directory
# - OPENCV_V: OpenCV version <X.X.X>
#
# ***

# Colors
RD='\033[0;31m' # Red
GR='\033[0;32m' # Green
BL='\033[0;34m' # Blue
NC='\033[0m'    # No Color

IN_DIR="${HOME}/src"
OPENCV_V="3.4.0"

# Check if sudo
if [ "$EUID" -eq 0 ]
  then printf "${RD}[!] Do not run as root (sudo).${NC}\n"
  exit
fi

clear
printf "${BL}[i] OpenCV ${OPENCV_V} environment init:${NC}\n"

# --- Install directory ---

printf "${BL}[i] (1/3) Making source dir '${IN_DIR}':${NC}\n"

mkdir ${IN_DIR}
cd ${IN_DIR}

# --- Download OpenCV source ---

printf "${BL}[i] (2/3) Dowloading OpenCV source v${OPENCV_V}:${NC}\n"

wget https://www.github.com/opencv/opencv/archive/${OPENCV_V}.zip -O opencv-${OPENCV_V}.zip
unzip opencv-${OPENCV_V}.zip

# --- Build OpenCV ---

printf "${BL}[i] (3/3) Creating Makefile:${NC}\n"

# CUDA_ARCH_BIN="6.2" for TX2, or "5.3" for TX1
cd ${IN_DIR}/opencv-${OPENCV_V}
mkdir build
cd build
cmake \
	-D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D WITH_CUDA=ON \
	-D CUDA_ARCH_BIN="6.2" \
	-D CUDA_ARCH_PTX="" \
	-D WITH_CUBLAS=ON \
	-D ENABLE_FAST_MATH=ON \
	-D CUDA_FAST_MATH=ON \
	-D ENABLE_NEON=ON \
	-D WITH_LIBV4L=ON \
	-D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D WITH_QT=ON \
	-D WITH_OPENGL=ON \
	..

# --- Info ---

printf "${BL}[i] To build and install OpenCV, run:${NC}\n"
printf "${GR}    cd ${PWD}${NC}\n"
printf "${GR}    make -j4${NC}\n"
printf "${GR}    make install${NC}\n"
