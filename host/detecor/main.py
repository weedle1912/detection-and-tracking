import numpy as np
import cv2
import os
import time

import detect
import ascii_art as art

#MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
LABEL_NAME = 'mscoco_label_map'
NUM_CLASSES = 90
VIDEO_FILE = '../../videos/BlueAngels.mp4'

def app():
    # Load model graph
    print 'Loading frozen model: ' + MODEL_NAME
    detection_graph = detect.load_tf_graph(MODEL_NAME)
    category_index = detect.get_label_index(LABEL_NAME, NUM_CLASSES)

    # Video source
    print 'Video source: ' + VIDEO_FILE
    cap = cv2.VideoCapture(VIDEO_FILE)

    # Run detecton
    print 'Running detection ...'
    detect.run_detection(cap, detection_graph, category_index)

    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    os.system('clear')
    art.printAsciiArt('Detection')
    print('v0.0.1 (C) weedle1912')
    app()