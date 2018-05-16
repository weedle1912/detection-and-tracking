import os
import math
import argparse

def main(args):
    f1 = open(args['file'][0], 'r')
    f2 = open(args['file'][1], 'r')
    file_out = open(args['output'], 'w')

    bboxes1 = []
    for line in f1:
        bboxes1.append(line_to_bbox(line))
    bboxes2 = []
    for line in f2:
        bboxes2.append(line_to_bbox(line))
    
    error_total = []
    n = min(len(bboxes1), len(bboxes2))
    for i in range(n):
        value = compare_center(bboxes1[i], bboxes2[i])
        error_total.append(value)
        file_out.write(str(value)+'\n')
    
    print('Avg. L2 error: %.2f'%(sum(error_total)/n*100))

    f1.close()
    f2.close()
    file_out.close()

def line_to_bbox(line):
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
    ap.add_argument('-f', '--file', required=True, nargs=2,
        help='path to csv files')
    ap.add_argument('-o', '--output', default='mc_out.csv',
        help='path to output file')
    
    args = vars(ap.parse_args())

    main(args)    
