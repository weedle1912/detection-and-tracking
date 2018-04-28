import cv2 # opencv-contrib-python required!
import threading
import sys
import time

if cv2.__version__ < '4.3.0':
    raise ImportError('Please upgrade your OpenCV installation to v4.3.* or later!')

class Tracker:
    def __init__(self, tracker_type='MEDIANFLOW'):
        self.tracker_type = tracker_type
        self.tracker = setTrackerType(self.tracker_type)
        self.bbox = (0,0,0,0)
        self.read_lock = threading.Lock()

    def init(self, frame, bbox):
        ok = self.tracker.init(frame, bbox)
        return ok

    def clear(self):
        self.tracker = setTrackerType(self.tracker_type)
    
    def update(self, frame):
        with self.read_lock:
            ok, self.bbox = self.tracker.update(frame)
    
    def get_bbox(self):
        with self.read_lock:
            bbox = self.bbox
        return bbox

def setTrackerType(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    else if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    else if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    else if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    else if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    else if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    else:
        tracker = None
    return tracker
