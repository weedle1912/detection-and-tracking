#!/bin/bash

# Store current directory
CWD=${PWD}

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
    if [ -f ${f}/${f}.mp4 ]; then
        echo ${f}/${f}.mp4 exists
    else
        mkdir ${f}
        python ${CWD}/../../scripts/images_to_video.py -d ${HOME}/Downloads/vot2017/${f}/ -o ${CWD}/${f}/${f}
        cp ${HOME}/Downloads/vot2017/${f}/groundtruth.txt ${CWD}/${f}
    fi
done <${CWD}/list.txt
