import cv2
from tracker import Tracker
from framehandler import FrameHandler

VIDEO_PATH = "../../videos/BlueAngels.mp4"

def main():
    handler = FrameHandler(VIDEO_PATH)
    tracker = Tracker()

    handler.start()
    handler.wait_new_frame()
    ok, frame = handler.read()
    tracker.init(frame, (60, 60, 130, 90))

    while True:
        handler.wait_new_frame()
        ok, frame = handler.read()
        if not ok:
            break
        tracker.update(frame)
        bbox = tracker.get_bbox()

        # Draw bbox
        draw_bbox(frame, bbox)
        cv2.imshow('frame', frame)
        if ( cv2.waitKey(1) & 0xFF == ord('q') ):
            break
    
    handler.stop()
    cv2.destroyAlleWindows()

def draw_bbox(frame, bbox):
    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)


if __name__=="__main__":
    main()
