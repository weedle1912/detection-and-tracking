import os
import math
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
    
    error_total = []
    n = min(len(bboxes_r), len(bboxes_t))
    for i in range(n):
        value = compare_center(bboxes_r[i], bboxes_t[i])
        error_total.append(value)
        file_out.write(str(value)+'\n')
    
    print('Avg. L2 error: %.2f'%(sum(error_total)/n*100))

    file_r.close()
    file_t.close()
    file_out.close()

def line_to_bbox(line, vid_size=FRAME_SIZE, frame_size=FRAME_SIZE):
    line = line.strip()
    if line == '()':
        return ()
    x, y, w, h = line.split(',')
    return (float(x), float(y), float(w), float(h))

def compare_center(b1, b2):
    if not b1 and not b2:
        return 0.00
    if not b1 or not b2:
        return 1.00
    cx1 = b1[0] + b1[2]//2
    cy1 = b1[1] + b1[3]//2
    cx2 = b2[0] + b2[2]//2
    cy2 = b2[1] + b2[3]//2
    e_x = abs(cx1-cx2)
    e_y = abs(cy1-cy2)
    error = math.sqrt(e_x**2 + e_y**2)

    return round( error , 4 )

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--result', required=True,
        help='path to result file')
    ap.add_argument('-t', '--truth', required=True,
        help='path to ground truth file')
    ap.add_argument('-o', '--output', default='mc_out.csv',
        help='path to output file')
    
    args = vars(ap.parse_args())

    main(args)    
