import cv2
import time
import argparse

BGR = [(255,0,0), (0,0,255), (0,255,0)]

def main(args):
    cap = cv2.VideoCapture(args['input'])
    width, height = args['size'][0], args['size'][1]
    series = []
    for i in range(len(args['file'])):
        bboxes = []
        f = open(args['file'][i], 'r')
        for line in f:
            line = line.strip()
            if not line == '()':
                x, y, w, h = line.strip().split(',')
                bbox = scale((float(x),float(y),float(w),float(h)), width, height)
            else:
                bbox = ()
            bboxes.append(bbox)
        f.close()
        series.append(bboxes)
    
    n = len(series[0])
    for s in series:
        n = min(n, len(s))

    if args['play']:
        val = 1
    else:
        val = 0

    for i in range(n):
        ok, frame = cap.read()
        if not ok: 
            break
        frame = cv2.resize(frame, (args['size'][0], args['size'][1]), interpolation=cv2.INTER_AREA)
        #cv2.rectangle(img,(xb,yb),(xb+wb,yb+hb),BGR[color],-1) 
        cv2.putText(frame,'Frame: %04d'%(i+1),(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1,cv2.LINE_AA)
        for j in range(len(series)):
            bbox = series[j][i]
            draw_bbox(frame, bbox, j)    
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(val) == 27: # Exit with 'esc' key
            break
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

def scale(bbox, width, height):
    x = int(bbox[0]*width)
    y = int(bbox[1]*height)
    w = int(bbox[2]*width)
    h = int(bbox[3]*height)
    return (x,y,w,h)

def draw_bbox(frame, bbox, color):
    if color > 2:
        color = 2
    if bbox:
        cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),BGR[color],2)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
        help='path to input video')
    ap.add_argument('-f', '--file', required=True, nargs='+',
        help='normalized csv bbox file(s)')
    ap.add_argument('-s', '--size', nargs=2, type=int, default=[640, 480],
        metavar=('WIDTH', 'HEIGHT'),
        help='frame size')
    ap.add_argument('-p', '--play', action='store_true',
        help='play video instead of stepping through each frame')
    args = vars(ap.parse_args())   

    main(args)
    