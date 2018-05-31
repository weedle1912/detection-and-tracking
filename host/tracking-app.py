import os
import cv2
import time
import argparse

# Modules
from src.videocapture import VideoCaptureAsync
from src.detector.detector import Detector
from src.tracker.tracker import Tracker

# Utilities
import src.utils.draw as draw_utils
import src.utils.bbox as bbox_utils
import src.utils.ascii_art as art_utils

MODEL_NAMES = [
    'ssd_inception_v2_coco_2017_11_17',
    'ssd_mobilenet_v2_coco_2018_03_29',
    'faster_rcnn_inception_v2_coco_2018_01_28'
]
LABEL_NAME = 'mscoco_label_map'
NUM_CLASSES = 90
TRACKER_TIMEOUT_SEC = 1.5

def run(args):
    cwd = os.getcwd()
    # Path to checkpoint (ckpt)
    model_path = os.path.join(cwd, 'src', 'detector', 'models', args['model'], 'frozen_inference_graph.pb')
    # Path to label names
    labels_path = os.path.join(cwd, 'src', 'detector', 'object_detection', 'data', args['label'] + '.pbtxt')

    print('[i] Init.')
    cap = VideoCaptureAsync(args['input'], args['size'][0], args['size'][1], args['fps'])
    detector = Detector(cap, model_path, labels_path, NUM_CLASSES)
    tracker = Tracker()
    ok, blank = cap.read()
    tracker.init(blank, (0,0,0,0))

    # Target class of interest
    target_class = args['target']
    target_id = detector.get_class_id(target_class)
    if not target_id:
        print('[!] Error: target class "%s" is not part of "%s".'%(args['target'], args['label']))
        exit()

    # Create out file
    if args['write']:
        # Video
        fourcc = cv2.VideoWriter_fourcc(*args['codec'])
        file_name = '%s%s'%(args['output'], args['ext'])
        out = cv2.VideoWriter(file_name, fourcc, args['fps'], (args['size'][0], args['size'][1]))
        # Comma Separated Values
        file_csv = open(args['output']+'.csv', 'w')

    detector.start()
    detector.wait() # First detection is slow
    cap.start()
    
    print('[i] Press "Esc" key to stop.')

    time_d = time.time()  # Time since last detection
    bbox_buffer = [()]*10 # For bbox stabilization

    while True:
        # Wait for new frame
        cap.wait()
        # Read frame
        ok, frame = cap.read(True)
        if not ok:
            break

        # Get detection
        new_detection, detections = detector.get_detections()
        bbox_d, score = bbox_utils.get_single_bbox(detections, target_id, args['size'][0], args['size'][1])
        # Filter detection on score
        if score < args['threshold']:
            bbox_d = ()

        # Tracker update
        if new_detection:
            if bbox_d:
                time_d = time.time()
                buffer = cap.read_frame_buffer()
                if buffer:
                    tracker = Tracker()
                    tracker.init(buffer.pop(0), bbox_d)
                    for f in buffer:
                        tracker.update(f)
            else:
                cap.clear_frame_buffer()
        tracker.update(frame)
        
        bbox_t = tracker.get_bbox()
        # Timeout tracker if detection lost
        if (time.time()-time_d > TRACKER_TIMEOUT_SEC):
            bbox_t = ()
        bbox_s, bbox_buffer = bbox_utils.bbox_stabilize(bbox_t, bbox_buffer)

        if not bbox_s:
            no_track = True
        else:
            no_track = False

        # Get FPS
        FPS_d = detector.get_fps()
        FPS_t = tracker.get_fps()
        # Check FPS dependency limit
        if int(FPS_t) > 0 and int(FPS_d) > 0:
            limit = args['fps']**2/FPS_d
            if FPS_t < limit:
                print('[!] Warning: FPS_t = '+('%d'%FPS_t).rjust(3)+' is too slow (limit = '+('%d).'%limit).rjust(3))

        # Frame overlay
        #draw_utils.draw_bbox(frame, bbox_d, target_class, 'green') # Detection - green
        #draw_utils.draw_bbox(frame, bbox_t, target_class, 'orange') # Tracking - orange
        draw_utils.draw_bbox(frame, bbox_s, target_class, 'red') # Stabilized - red
        draw_utils.draw_header(frame, target_class, score)
        draw_utils.draw_footer(frame, FPS_d, FPS_t, no_track, args['size'][1])

        # Display frame
        if args['write']:
            # Video
            out.write(frame)
            # CSV (normalized bbox)
            line = make_csv_line(bbox_s, args['size'][0], args['size'][1])
            file_csv.write(line)
        else:
            cv2.imshow('Frame: %dx%d, %.1f FPS'%(args['size'][0],args['size'][1], args['fps']), frame)
            if cv2.waitKey(1) == 27: # Exit with 'esc' key
                break
    
    detector.stop()
    cap.stop()
    if args['write']:
        out.release()
        file_csv.close()
    cv2.destroyAllWindows()   

def print_settings(args):
    print('--- Source ---')
    print('* Input:     ' + str(args['input']))
    print('* Size:      %dx%d'%(args['size'][0], args['size'][1]))
    print('* FPS:       %.1f'%args['fps'])
    print('--- Detector ---')
    print('* Model:     ' + str(args['model']))
    print('* Labels:    ' + str(args['label']))
    print('--- Object ---')
    print('* Target:    ' + str(args['target']))
    print('* Threshold: ' + str(args['threshold'])+'%')
    if args['write']:
        print('--- Output ---')
        print('* File:      ' + str(args['output']) + str(args['ext']))
        print('* Codec:     ' + str(args['codec']))
    print

def make_csv_line(bbox, width, height):
    bbox_n = bbox_utils.bbox_normalize(bbox, width, height, 4)
    if bbox_n:
        line = str(bbox_n[0])+','+str(bbox_n[1])+','+str(bbox_n[2])+','+str(bbox_n[3])+'\n'
    else:
        line = '()\n'
    return line

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default='../videos/HobbyKing.mp4',
        metavar='SRC',
        help='path to video source')
    ap.add_argument('-t', '--target', default='airplane',
        metavar='TARGET_CLASS',
        help='target class to track')
    ap.add_argument('-th', '--threshold', type=int, default=50,
        metavar='SCORE',
        help='detection score threshold (0-100)')
    ap.add_argument('-s', '--size', nargs=2, type=int, default=[640,480], 
        metavar=('WIDTH', 'HEIGHT'),
        help='video frame size')
    ap.add_argument('-f', '--fps', type=float, default=30,
        help='video playback rate')
    ap.add_argument('-m', '--model', default='ssd_mobilenet_v2_coco_2018_03_29',
        metavar='MODEL_NAME',
        help='name of inference model')
    ap.add_argument('-l', '--label', default='mscoco_label_map',
        metavar='LABEL_NAME',
        help='name of label file')
    ap.add_argument('-w', '--write', action='store_true',
        help='wether to write results to file')
    ap.add_argument('-o', '--output', default='out',
        metavar='FILE_NAME',
        help='name of output file (w/o ext)')
    ap.add_argument('-c', '--codec', default='mp4v',
        metavar='FOURCC',
        help='fourcc codec for output file')
    ap.add_argument('-e', '--ext', default='.mp4',
        help='ext (container) for output file')
    args = vars(ap.parse_args())
    # Run
    os.system('clear')
    art_utils.printAsciiArt('Autonomous')
    art_utils.printAsciiArt('Tracking')
    print('tracking-app v1.0.0 (C) weedle1912\n')
    print_settings(args)
    print('--- Running app:')
    run(args)