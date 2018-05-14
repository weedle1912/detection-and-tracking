import cv2
import time
import argparse

def main(args):
    cap = cv2.VideoCapture(args['input'])
    f = open(args['file'], 'r')
    width, height = args['size'][0], args['size'][1]
    bboxes = []
    for line in f:
        x, y, w, h = line.strip().split(',')
        x = int(float(x)*width)
        y = int(float(y)*height)
        w = int(float(w)*width)
        h = int(float(h)*height)
        bboxes.append((x,y,w,h))
    f.close()
    print('Count: %d frames'%len(bboxes))

    for bbox in bboxes:
        ok, frame = cap.read()
        if not ok: 
            break
        frame = cv2.resize(frame, (args['size'][0], args['size'][1]), interpolation=cv2.INTER_AREA)
        if bbox:
            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27: # Exit with 'esc' key
            break
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
        help='path to input video')
    ap.add_argument('-f', '--file', required=True,
        help='input gt file')
    ap.add_argument('-s', '--size', nargs=2, type=int, default=[640, 480],
        metavar=('WIDTH', 'HEIGHT'),
        help='frame size')
    args = vars(ap.parse_args())   

    main(args)