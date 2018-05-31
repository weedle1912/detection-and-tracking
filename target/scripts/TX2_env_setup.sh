#!/bin/bash
# Colors
RD='\033[0;31m' # Red
GR='\033[0;32m' # Green
BL='\033[0;34m' # Blue
NC='\033[0m'    # No Color

# Path
ENV_NAME='tx2-env'

# Check if sudo
if [ "$EUID" -ne 0 ]
  then printf "${RD}[!] Please run script as root (sudo).${NC}\n"
  exit
fi

clear
printf "${BL}[i] Setting up virtual environment '${ENV_NAME}':${NC}\n"

# -- Dependencies --

printf "${BL}[i] (1/3) Installing dependencies:${NC}\n" 

apt-get install python-pip python-dev python-virtualenv

# -- Environment --

printf "${BL}[i] (2/3) Setting up environment in ~/${ENV_NAME}:${NC}\n"

mkdir ~/${ENV_NAME}
virtualenv ~/${ENV_NAME}
# Activate
source ~/${ENV_NAME}/bin/activate

# -- Requirements --

printf "${BL}[i] (3/3) Installing requirements:${NC}\n"
if [ -e requirements.txt ]
then
    pip install -r requirements.txt
else
    printf "${RD}[!] Requirements not installed: File not found.${NC}\n"
fi

# -- Info --
printf "${BL}[i] Activate virtual environment:\n"
printf "${GR}    source ~/${ENV_NAME}/bin/activate${NC}\n"
printf "${BL}[i] Deactivate virtual environment:${NC}\n"
printf "${GR}    deactivate${NC}\n"
