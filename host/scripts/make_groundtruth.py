import cv2
import os
import argparse

def main(args):
    cap = cv2.VideoCapture(args['input'])
    n = 0
    # If out file exist, continue at checkpoint
    if os.path.isfile(args['output']):
        out = open(args['output'], 'r')
        n = len(out.readlines())
        out.close()
        out = open(args['output'], 'a')
    else:
        out = open(args['output'], 'w')

    if args['limit']:
        limit = args['limit'] + n
    frame_count = n

    # Continue at next frame
    ok = skip_frames(cap, frame_count)
    
    # Loop through video
    while ok:
        frame_count += 1
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (args['size'][0], args['size'][1]), interpolation=cv2.INTER_AREA)

        # Select bounding box in frame
        win_name = 'Frame: %d'%frame_count
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 20,20)
        bbox = cv2.selectROI(win_name, frame, False)
        cv2.destroyWindow(win_name)

        # Normalize coordinates
        bbox = bbox_normalize(bbox, args['size'])
        line = str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3])
        print('bbox: %s'%line)
        out.write(line+'\n')

        if args['limit']:
            if frame_count == limit:
                break

    cap.release()
    out.close()

def skip_frames(cap, n):
    ok = True
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
    return ok

def bbox_normalize(bbox, size, precision=4):
    width, height = size[0], size[1]
    x = round( float(bbox[0])/width, precision )
    y = round( float(bbox[1])/height, precision )
    w = round( float(bbox[2])/width, precision )
    h = round( float(bbox[3])/height, precision )
    return(x,y,w,h)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
        help='path to input video')
    ap.add_argument('-o', '--output', required=True,
        help='path to output text file')
    ap.add_argument('-s', '--size', nargs=2, type=int, default=[640, 480],
        metavar=('WIDTH', 'HEIGHT'),
        help='frame size')
    ap.add_argument('-l', '--limit', type=int, default=0,
        help='limit of number of frames to process')
    args = vars(ap.parse_args())   

    main(args)