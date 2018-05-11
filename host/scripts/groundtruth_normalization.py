import argparse

def main(args):
    width, height = args['size']
    f_in = open(args['input'], 'r')
    f_out = open(args['output'], 'w')
    for line in f_in:
        stripped = ''.join(c for c in line if c in ',.0123456789')
        x, y, w, h = stripped.split(',')
        f_out.write( '%.4f,%.4f,%.4f,%.4f\n'%(float(x)/width, float(y)/height, float(w)/width, float(h)/height) )
    f_in.close()
    f_out.close()

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
        help='path to input file')
    ap.add_argument('-s', '--size', nargs=2, type=int, required=True,
        metavar=('WIDTH', 'HEIGHT'),
        help='frame size of video')
    ap.add_argument('-o', '--output', default='ground_norm.csv',
        help='path to output file')
    
    args = vars(ap.parse_args())
    main(args) 