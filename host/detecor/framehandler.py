# *******************************************************
# * Module: Frame Handler
# * Inspired by repo: gilbertfrancois/video-capture-async
# *
# *******************************************************

import cv2
import threading
import time

class FrameHandler:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.new_frame = threading.Event()
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)
    
    def get(self, var):
        return self.cap.get(var)

    def start(self):
        if self.started:
            print('[!] FrameCapture thread already started.')
            return None
        self.started = True
        self.thread = threading.Thread(name='Frame handler', target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.new_frame.set()
            # Sleep depend on FPS
            time.sleep(1/30.0)
    
    def wait_new_frame(self):
        self.new_frame.wait()

    def read(self):
        with self.read_lock:
            if self.grabbed:
                frame = self.frame.copy()
                grabbed = True
            else:
                frame = None
                grabbed = False
            self.new_frame.clear()
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()