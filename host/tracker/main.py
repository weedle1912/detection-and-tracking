import cv2
from tracker import Tracker

VIDEO_PATH = "../../videos/HobbyKing.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    tracker = Tracker()
    isInit = False
    bbox = (474,469,143,62)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        if not isInit:
            tracker.init(frame, bbox)
            isInit = True
        else:
            tracker.update(frame)
        
        bbox = tracker.get_bbox()

        draw_bbox(frame, bbox)
        cv2.imshow('frame', frame)
        if ( cv2.waitKey(1) & 0xFF == ord('q') ):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def draw_bbox(frame, bbox):
    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)


if __name__=="__main__":
    main()
