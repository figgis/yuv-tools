#!/usr/bin/env python

import sys
from pylab import *
import matplotlib.pyplot as plt
import numpy
from ycbcr import YCbCr

def load_data(yuv, f):
    for luma in yuv.get_luma(f): break
    n = numpy.array(luma, dtype=numpy.uint8)
    img = numpy.reshape(n, (Y.height, Y.width))

    return img

def usage(me):
    """
    """
    print "%s filename1, filename2 width height format" % me
    sys.exit(0)

if __name__ == '__main__':

    FMT = ['IYUV', 'UYVY', 'YV12', 'YVYU']

    if len(sys.argv) != 6:
        usage(sys.argv[0])

    fname1 = sys.argv[1]
    fname2 = sys.argv[2]
    w = int(sys.argv[3])
    h = int(sys.argv[4])
    if sys.argv[5] in FMT:
        fmt = sys.argv[5]
    else:
        usage(sys.argv[0])

    Y = YCbCr(width=w, height=h, filename='foreman_cif_frame_0.yuv',
              yuv_format_in='YV12', filename_diff='foreman_cif_frame_1.yuv')

    # First subplot
    figure()
    frame1 = subplot(121)
    plt.imshow(load_data(Y, fname1), cmap=cm.gray, hold=True)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    # Second subplot
    frame2 = subplot(122)
    plt.imshow(load_data(Y, fname2), cmap=cm.gray, hold=True)
    frame2.axes.get_xaxis().set_visible(False)
    frame2.axes.get_yaxis().set_visible(False)

    plt.suptitle('PSNR = %f' % Y.psnr()[0], fontsize=14)
    plt.show()
