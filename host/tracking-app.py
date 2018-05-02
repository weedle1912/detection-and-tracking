import os
import cv2
import time

from detector.detector import Detector
from detector.videocapture import VideoCaptureAsync
from tracker.tracker import Tracker

MODEL_NAMES = [
    'ssd_inception_v2_coco_2017_11_17',
    'ssd_mobilenet_v2_coco_2018_03_29',
    'faster_rcnn_inception_v2_coco_2018_01_28'
]
LABEL_NAME = 'mscoco_label_map'
NUM_CLASSES = 90
VIDEO_FILE = '../videos/HobbyKing.mp4'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

def test():
    cwd = os.getcwd()
    # Path to checkpoint (ckpt)
    model_path = os.path.join(cwd, 'detector', 'models', MODEL_NAMES[1], 'frozen_inference_graph.pb')
    # Path to label names
    labels_path = os.path.join(cwd, 'detector', 'object_detection', 'data', LABEL_NAME + '.pbtxt')

    print('[i] Init.')
    cap = VideoCaptureAsync(VIDEO_FILE, FRAME_WIDTH, FRAME_HEIGHT, FPS)
    detector = Detector(cap, model_path, labels_path, NUM_CLASSES)
    tracker = Tracker()
    init = False  

    detector.start()
    detector.wait() # First detection is slow
    cap.start()

    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        new, detections = detector.get_detections()
        bbox_d = get_track_coord(detections)
        draw_bbox(frame, bbox_d, (0,255,0)) # Detection - green

        if (not init) or new:        
            tracker = Tracker()
            tracker.init(frame, bbox_d)
            init = True
        else:
            tracker.update(frame)

        bbox_t = tracker.get_bbox()

        draw_bbox(frame, bbox_t, (0,153,255)) # Tracking - orange
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == 27:
            break
        
        count += 1
    
    detector.stop()
    cap.stop()
    cv2.destroyAllWindows()

def draw_detections(img, det_dict, n):
    for i in range(n):
        #[ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = det_dict['detection_boxes'][i]
        bbox = [(int(xmin*FRAME_WIDTH), int(ymin*FRAME_HEIGHT)), (int(xmax*FRAME_WIDTH), int(ymax*FRAME_HEIGHT))]
        cv2.rectangle(img,bbox[0],bbox[1],(255,0,0),2)

def draw_bbox(frame, bbox, color):
    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color,2)

def get_track_coord(det_dict):
    ymin, xmin, ymax, xmax = det_dict['detection_boxes'][0]
    x = int(xmin*FRAME_WIDTH)
    y = int(ymin*FRAME_HEIGHT)
    w = int(xmax*FRAME_WIDTH) - int(xmin*FRAME_WIDTH)
    h = int(ymax*FRAME_HEIGHT) - int(ymin*FRAME_HEIGHT)
    return (x,y,w,h)

if __name__ == '__main__':
    test()