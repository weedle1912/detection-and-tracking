#!/bin/bash

# Store current directory
CWD=${PWD}
mkdir vot2017

# Check if downloaded
if ! [ -f ${HOME}/Downloads/vot2017.zip ]; then
    cd ${HOME}/Downloads
    wget http://data.votchallenge.net/vot2017/vot2017.zip
fi
# Check if unzipped
if ! [ -f ${HOME}/Downloads/vot2017/list.txt ]; then
    cd ${HOME}/Downloads
    unzip vot2017.zip -d ./vot2017
fi
cd ${CWD}

# Construct videos from images in folders
while read f; do
    echo Constructing video: ${f}
    if [ -f vot2017/${f}/${f}.mp4 ]; then
        echo vot2017/${f}/${f}.mp4 exists
    else
        mkdir vot2017/${f}
        python ../scripts/images_to_video.py -d ${HOME}/Downloads/vot2017/${f}/ -o vot2017/${f}/${f}
        cp ${HOME}/Downloads/vot2017/${f}/groundtruth.txt ./vot2017/${f}
    fi
done <${HOME}/Downloads/vot2017/list.txt