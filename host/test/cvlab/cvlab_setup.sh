#!/bin/bash

# Store current directory
CWD=${PWD}
TMP_DIR='tmp'
mkdir ${TMP_DIR}

# Go through list
while read f; do
    echo Video: ${f}
    if [ -f ${f}/${f}.mp4 ]; then
        echo ${f}/${f}.mp4 exists
    else
        mkdir ${f}
        cd ${TMP_DIR}
        url='http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/'${f}'.zip'
        wget ${url}
        unzip ${f}.zip -d ${f}
        python ${CWD}/../../scripts/images_to_video.py -d ${f}/${f}/img/ -o ${CWD}/${f}/${f}
        cp ${f}/${f}/groundtruth_rect.txt ${CWD}/${f}/
        cd ${CWD}
    fi
done <${CWD}/list.txt

# Remove tmp dir (zip files and images)
rm -r ${TMP_DIR}
