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
        self.grabbed = True
        self.frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.frame_buffer = []
        self.started = False
        self.new_frame = threading.Event()
        self.read_lock = threading.Lock()
        self.thread_capture = threading.Thread(name='Video Capture', target=self.update, args=())
        # For timing
        self.fps = fps
        self.timer = threading.Event()
        self.thread_timer = threading.Thread(name='Capture Timer', target=self.time_loop, args=())

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
        self.thread_capture.start()
        return self

    def update(self):
        self.thread_timer.start()
        while self.started:
            self.timer.clear()
            grabbed, frame = self.cap.read()
            if grabbed:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            with self.read_lock:
                # Old frame in buffer
                self.frame_buffer.append(self.frame)
                # Update with new frame
                self.frame = frame
                self.grabbed = grabbed
            self.new_frame.set()
            # Wait for next time tick
            self.timer.wait(1)
        
        self.thread_timer.join()
    
    def time_loop(self):
        while self.started:
            time.sleep(1.0/self.fps)
            self.timer.set()

    def wait(self):
        self.new_frame.wait()

    def read(self, clear_event=False):
        with self.read_lock:
            if self.grabbed:
                frame = self.frame.copy()
                grabbed = True
            else:
                frame = None
                grabbed = False
        if clear_event:
            self.new_frame.clear()
        return grabbed, frame

    def read_frame_buffer(self):
        with self.read_lock:
            buffer = copy.deepcopy(self.frame_buffer)
            self.frame_buffer = []
        return buffer
    
    def clear_frame_buffer(self):
        with self.read_lock:
            self.frame_buffer = []

    def stop(self):
        print('[c] Stopping.')
        self.started = False
        self.thread_capture.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()