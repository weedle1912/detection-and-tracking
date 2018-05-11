import os
import cv2
import argparse

FRAME_SIZE = (640,480)
VID_SIZE = (1280,720)

def main(args):
    file_r = open(args['result'], 'r')
    file_t = open(args['truth'], 'r')
    file_out = open(args['output'], 'w')

    bboxes_r = []
    for line in file_r:
        bboxes_r.append(line_to_bbox(line))
    bboxes_t = []
    for line in file_t:
        bboxes_t.append(line_to_bbox(line, VID_SIZE))
    
    iou_total = []
    n = min(len(bboxes_r), len(bboxes_t))
    for i in range(n):
        value = iou(bboxes_r[i], bboxes_t[i])
        iou_total.append(value)
        file_out.write(str(value)+'\n')
    
    print('Avg. iou: %.2f'%(sum(iou_total)/n*100))

    file_r.close()
    file_t.close()
    file_out.close()

def line_to_bbox(line, vid_size=FRAME_SIZE, frame_size=FRAME_SIZE):
    line = line.strip()
    if line == '()':
        return ()
    x, y, w, h = line.split(',')
    return (float(x), float(y), float(w), float(h))

def iou(b1, b2):
    if not b1 and not b2:
        return 1.00
    if not b1 or not b2:
        return 0.00
    xmin1, ymin1, xmax1, ymax1 = b1[0], b1[1], b1[0]+b1[2], b1[1]+b1[3]
    xmin2, ymin2, xmax2, ymax2 = b2[0], b2[1], b2[0]+b2[2], b2[1]+b2[3]
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)
    a_iw = x2-x1
    a_ih = y2-y1
    if a_iw < 0 or a_ih < 0:
        return 0.00
    a_i = a_iw*a_ih
    a_u = b1[2]*b1[3] + b2[2]*b2[3] - a_i

    return round( float(a_i)/a_u , 2 )

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--result', required=True,
        help='path to result file')
    ap.add_argument('-t', '--truth', required=True,
        help='path to ground truth file')
    ap.add_argument('-o', '--output', default='iou_out.csv',
        help='path to output file')
    
    args = vars(ap.parse_args())

    main(args)    
