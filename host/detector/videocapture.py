# *******************************************************
# * Module: Video Capture
# * Inspired by repo: gilbertfrancois/video-capture-async
# *
# *******************************************************

import numpy as np
import cv2
import threading
import time
import copy

class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.width = width
        self.height = height
        self.fps = fps
        self.grabbed = True
        self.frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.frame_buffer = []
        self.started = False
        self.new_frame = True
        self.buffer_lock = threading.Lock()
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)
    
    def get(self, var):
        return self.cap.get(var)

    def start(self):
        if self.started:
            print('[!] Video capture already started.')
            return None
        print('[c] Starting.')
        self.started = True
        self.thread = threading.Thread(name='Video Capture', target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.new_frame = True
            with self.buffer_lock:
                self.frame_buffer.append(frame)
            # Sleep depend on FPS
            time.sleep(1.0/self.fps)
    
    def isNewframe(self):
        return self.new_frame

    def read(self):
        with self.read_lock:
            if self.grabbed:
                frame = self.frame.copy()
                grabbed = True
            else:
                frame = None
                grabbed = False
            self.new_frame = False
        return grabbed, frame

    def read_frame_buffer(self):
        with self.buffer_lock:
            buffer = copy.deepcopy(self.frame_buffer)
            self.frame_buffer = []
        return buffer

    def stop(self):
        print('[c] Stopping.')
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()