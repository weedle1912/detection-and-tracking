import os
import cv2
import time
import argparse

import detector.ascii_art as art 

from detector.detector import Detector
from detector.videocapture import VideoCaptureAsync
from tracker.tracker import Tracker

MODEL_NAMES = [
    'ssd_inception_v2_coco_2017_11_17',
    'ssd_mobilenet_v2_coco_2018_03_29',
    'faster_rcnn_inception_v2_coco_2018_01_28'
]
LABEL_NAME = 'mscoco_label_map'
CLS_NAMES = {'airplane':5}
NUM_CLASSES = 90
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

BGR = {'green':(0,255,0), 'orange':(0,153,255), 'white':(255,255,255), 'red':(0,0,255), 'black':(0,0,0)}

bbox_buffer = [()]*10

def test(args):
    cwd = os.getcwd()
    # Path to checkpoint (ckpt)
    model_path = os.path.join(cwd, 'detector', 'models', args['model'], 'frozen_inference_graph.pb')
    # Path to label names
    labels_path = os.path.join(cwd, 'detector', 'object_detection', 'data', args['label'] + '.pbtxt')

    print('[i] Init.')
    print('[i] Source: %s'%args['input'])
    cap = VideoCaptureAsync(args['input'], FRAME_WIDTH, FRAME_HEIGHT, FPS)
    detector = Detector(cap, model_path, labels_path, NUM_CLASSES)
    tracker = Tracker()

    # Create out file
    if args['write']:
        fourcc = cv2.VideoWriter_fourcc(*args['codec'])
        file_name = '%s%s'%(args['output'], args['ext'])
        print('[i] Output: %s (c: %s)'%(file_name, args['codec']))
        out = cv2.VideoWriter(file_name, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    detector.start()
    detector.wait() # First detection is slow
    cap.start()

    t_start = False

    while True:
        # Wait for new frame
        cap.wait()
        # Read frame
        ok, frame = cap.read(True)
        if not ok:
            break

        # Get detection
        new_detection, detections = detector.get_detections()
        bbox_d = get_single_bbox(detections, CLS_NAMES['airplane'])

        if not tracker.isInit():
            tracker = Tracker()
            tracker.init(frame, (0,0,0,0))

        # Tracker update
        if new_detection:
            if bbox_d:
                buffer = cap.read_frame_buffer()
                if buffer:
                    tracker = Tracker()
                    tracker.init(buffer.pop(0), bbox_d)
                    for f in buffer:
                        tracker.update(f)
                    tracker.update(frame)
            else:
                cap.clear_frame_buffer()
        else:
            tracker.update(frame)
        
        bbox_t = tracker.get_bbox()
        bbox_s = stabilize(bbox_t)

        if not bbox_s:
            no_track = True
        else:
            no_track = False

        # Get FPS
        FPS_d = detections['FPS']
        FPS_t = tracker.get_fps()

        # Frame overlay
        #draw_bbox(frame, bbox_d, BGR['green']) # Detection - green
        #draw_bbox(frame, bbox_t, BGR['orange']) # Tracking - orange
        draw_bbox(frame, bbox_s, BGR['red']) # Stabilized - red
        draw_header(frame)
        draw_footer(frame, FPS_d, FPS_t, no_track)

        # Display frame
        if args['write']:
            out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == 27:
            break
    
    detector.stop()
    cap.stop()
    if args['write']:
        out.release()
    cv2.destroyAllWindows()

def stabilize(bbox):
    # Update buffer (FILO)
    if bbox not in bbox_buffer:
        bbox_buffer.pop(0)
        bbox_buffer.append(bbox)
    if not bbox:
        return ()
    
    # Save box center
    cx, cy = bbox[0]+(bbox[2]//2), bbox[1]+(bbox[3]//2)
    ws,hs = [],[]
    for b in bbox_buffer:
        if b:
            ws.append(b[2])
            hs.append(b[3])
    if not ws or not hs:
        return ()
    
    # Calc median size
    ws.sort()
    hs.sort()
    w = ws[len(ws)//2]
    h = hs[len(ws)//2]
    # Increase size with 20%
    w = int(w*1.2)
    h = int(h*1.2)
    x = cx - (w//2)
    y = cy - (h//2)
    return (x,y,w,h)


def draw_bbox(frame, bbox, color):
    if bbox:
        cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color,2)
        draw_label(frame, bbox, 'airplane', color)

def draw_label(img, bbox, label, color):
    # Calc position
    s, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1) 
    xb, yb, wb, hb = bbox[0]-1, bbox[1]-s[1]-6, s[0]+2, s[1]+6
    xl, yl = bbox[0], bbox[1]-3
    if yl < 13:
        yb += (bbox[3]+hb)
        yl += (bbox[3]+hb)

    # Draw background and text
    cv2.rectangle(img,(xb,yb),(xb+wb,yb+hb),color,-1) 
    cv2.putText(img,label,(xl,yl), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)

def draw_header(img):
    cv2.putText(img,'Class: Airplane',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,BGR['black'],1,cv2.LINE_AA)

def draw_footer(img, fps_d, fps_t, no_track): 
    cv2.putText(img,( 'Det. FPS: ' + ('%d'%fps_d).rjust(3) ),(10,FRAME_HEIGHT-25), cv2.FONT_HERSHEY_PLAIN, 1,BGR['black'],1,cv2.LINE_AA)
    if no_track:
        cv2.putText(img,( 'No track.' ),(10,FRAME_HEIGHT-10), cv2.FONT_HERSHEY_PLAIN, 1,BGR['red'],1,cv2.LINE_AA)
    else:
        cv2.putText(img,( 'Trc. FPS: ' + ('%d'%fps_t).rjust(3) ),(10,FRAME_HEIGHT-10), cv2.FONT_HERSHEY_PLAIN, 1,BGR['black'],1,cv2.LINE_AA)

def get_single_bbox(det_dict, class_index):
    bboxes, scores = [], []
    for i in range(det_dict['num_detections']):
        if det_dict['detection_classes'][i] == class_index:
            return format_bbox(det_dict['detection_boxes'][i])
            scores.append(det_dict['detection_scores'][i])
    return ()

def format_bbox(bbox_norm):
    ymin, xmin, ymax, xmax = bbox_norm
    x = int(xmin*FRAME_WIDTH)
    y = int(ymin*FRAME_HEIGHT)
    w = int(xmax*FRAME_WIDTH) - int(xmin*FRAME_WIDTH)
    h = int(ymax*FRAME_HEIGHT) - int(ymin*FRAME_HEIGHT)
    return (x,y,w,h)    

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default='../videos/HobbyKing.mp4',
        help='path to video source')
    ap.add_argument('-m', '--model', default='ssd_mobilenet_v2_coco_2018_03_29',
        help='name of inference model')
    ap.add_argument('-l', '--label', default='mscoco_label_map',
        help='name of label file')
    ap.add_argument('-w', '--write', action='store_true',
        help='wether or not to write result to file')
    ap.add_argument('-o', '--output', default='out',
        help='name of output file (w/o ext)')
    ap.add_argument('-c', '--codec', default='mp4v',
        help='fourcc coded for output file')
    ap.add_argument('-e', '--ext', default='.mp4',
        help='ext (container) for output file')
    args = vars(ap.parse_args())
    # Run
    os.system('clear')
    art.printAsciiArt('Tracking')
    print('\n*****************************')
    print('Tracker v0.0.1 (c) weedle1912')
    print('*****************************\n')
    test(args)