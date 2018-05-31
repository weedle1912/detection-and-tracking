#!/bin/bash

# ***
# Remove all previous installations of OpenCV
# and install necessary dependencies.
#
# ***

# Colors
RD='\033[0;31m' # Red
BL='\033[0;34m' # Blue
NC='\033[0m'    # No Color

# Check if sudo
if [ "$EUID" -ne 0 ]
  then printf "${RD}[!] Please run as root (sudo).${NC}\n"
  exit
fi

clear
printf "${BL}[i] OpenCV install preparations:${NC}\n"

# --- Remove old versions ---

printf "${BL}[i] Removing old versions:${NC}\n"

# Remove all OpenCV files installed by JetPack (OpenCV4Tegra)
apt-get purge libopencv*
# Remove apt-get version of numpy
apt-get purge python-numpy
# Clean up unused apt packages
apt autoremove
# Upgrade all apt packages to latest version (optional)
#apt-get update
#apt-get dist-upgrade
# Update gcc apt package to the latest version (highly recommended)
printf "${BL}[i] Updating compiler:${NC}\n"
apt-get install -y --only-upgrade g++-5 cpp-5 gcc-5

# --- Install dependencies ---

printf "${BL}[i] Installing relevant dependencies:${NC}\n"

# Install dependencies based on the Jetson "Installing OpenCV Guide"
printf "${BL}[i] (1/5) From Jetson OpenCV Guide:${NC}\n"
apt-get install -y build-essential make cmake cmake-curses-gui \
	g++ libavformat-dev libavutil-dev \
	libswscale-dev libv4l-dev libeigen3-dev \
	libglew-dev libgtk2.0-dev

# Install dependencies for GStreamer
printf "${BL}[i] (2/5) For GStreamer:${NC}\n"
apt-get install -y libdc1394-22-dev libxine2-dev \
	libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev

# Install additional dependencies based on pyimageresearch article
printf "${BL}[i] (3/5) For python imaging:${NC}\n"
apt-get install -y libjpeg8-dev libjpeg-turbo8-dev libtiff5-dev \
	libjasper-dev libpng12-dev libavcodec-dev
apt-get install -y libxvidcore-dev libx264-dev libgtk-3-dev \
	libatlas-base-dev gfortran

# Install Qt5 dependencies (needed for OpenGL support)
printf "${BL}[i] (4/5) For OpenGL:${NC}\n"
apt-get install -y qt5-default

# Install dependencies for python2
printf "${BL}[i] (5/5) For python 2:${NC}\n"
sudo apt-get install -y python-dev python-pip python-tk
sudo pip install numpy

# Install dependencies for python 3 (optional)
#sudo apt-get install python3-dev python3-pip python3-tk
#sudo pip3 install numpy

