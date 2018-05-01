import numpy as np
import cv2
import os
import time

import detect
from detector import Detector
from framehandler import FrameHandler
from videocapture import VideoCaptureAsync
import ascii_art as art

MODEL_NAMES = [
    'ssd_inception_v2_coco_2017_11_17',
    'ssd_mobilenet_v2_coco_2018_03_29',
    'faster_rcnn_inception_v2_coco_2018_01_28'
]
LABEL_NAME = 'mscoco_label_map'
NUM_CLASSES = 90
VIDEO_FILE = '../../videos/HobbyKing.mp4'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def app():
    print('[i] Init.')
    cap = VideoCaptureAsync(VIDEO_FILE, FRAME_WIDTH, FRAME_HEIGHT)
    detector = Detector(cap, MODEL_NAMES[1], LABEL_NAME, NUM_CLASSES)

    cap.start()
    detector.start()
    
    bboxes = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        bboxes_new = detector.get_bboxes()
        if bboxes_new:
            bboxes = bboxes_new

        for i in range(len(bboxes)):
            cv2.rectangle(frame,bboxes[i][0],bboxes[i][1],(255,0,0),2)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == 27:
            break
    
    detector.stop()
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.system('clear')
    art.printAsciiArt('Detection')
    print('v1.0.0 (C) weedle1912')
    app()
