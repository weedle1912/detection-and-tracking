#!/bin/bash
# Colors
RD='\033[0;31m' # Red
GR='\033[0;32m' # Green
BL='\033[0;34m' # Blue
NC='\033[0m'    # No Color

# Path
ENV_NAME='tensorflow'

clear
printf "${BL}[i] Setting up virtual environment '${ENV_NAME}':${NC}\n"

# -- Environment --

printf "${BL}[i] (1/3) Setting up environment in ~/${ENV_NAME}:${NC}\n"

mkdir ~/${ENV_NAME}
virtualenv ~/${ENV_NAME}
# Activate
source ~/${ENV_NAME}/bin/activate

# -- Requirements --

printf "${BL}[i] (2/3) Installing requirements:${NC}\n"
if [ -e requirements.txt ]
then
    pip install -r requirements.txt
else
    printf "${RD}[!] Requirements not installed: File not found.${NC}\n"
fi

# -- Object Detection API (TensorFlow) --
printf "${BL}[i] (3/3) Downloading TF Object Detection API:${NC}\n"
mkdir tmp
git clone https://github.com/tensorflow/models ./tmp/models
cp -r tmp/models/research/object_detection/* src/detector/object_detection/
rm -r tmp

# -- Info --
printf "${BL}[i] Activate virtual environment:\n"
printf "${GR}    source ~/${ENV_NAME}/bin/activate${NC}\n"
printf "${BL}[i] Deactivate virtual environment:${NC}\n"
printf "${GR}    deactivate${NC}\n"
