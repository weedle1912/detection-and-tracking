import os
import cv2
import argparse

FPS = 30

def main(args):
    images = [f for f in os.listdir(args['dir']) if f.endswith(args['type'])]
    images.sort()
    height, width, _ =  cv2.imread(args['dir'] + images[0]).shape

    # Create video file
    fourcc = cv2.VideoWriter_fourcc(*args['codec'])
    file_name = '%s%s'%(args['output'], args['ext'])
    print('[i] Video file: %s (c: %s)'%(file_name, args['codec']))
    out = cv2.VideoWriter(file_name, fourcc, FPS, (width, height))

    # Write images to video file
    for i in images:
        img = cv2.imread(args['dir'] + i)
        out.write(img)

    out.release()

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dir', required=True,
        help='path to folder of images')
    ap.add_argument('-t', '--type', default='.jpg',
        help='image ext/type')
    ap.add_argument('-o', '--output', default='video',
        help='name of output file (w/o ext)')
    ap.add_argument('-c', '--codec', default='mp4v',
        help='fourcc codec of video file')
    ap.add_argument('-e', '--ext', default='.mp4',
        help='ext (container) for video file')
    
    args = vars(ap.parse_args())

    main(args)    
