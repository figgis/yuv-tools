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

    Y = YCbCr(width=w, height=h, filename=fname1,
              yuv_format_in=fmt, filename_diff=fname2)

    data1 = load_data(Y, fname1)
    data2 = load_data(Y, fname2)

    psnr = Y.psnr().next()[0]
    ssim = Y.ssim().next()

    # First subplot
    figure()
    frame1 = subplot(121)
    plt.imshow(data1, cmap=cm.gray, hold=True)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    title("PSNR = %f" % psnr)

    # Second subplot
    frame2 = subplot(122)
    plt.imshow(data2, cmap=cm.gray, hold=True)
    frame2.axes.get_xaxis().set_visible(False)
    frame2.axes.get_yaxis().set_visible(False)
    title("SSIM = %f" % ssim)

    plt.show()
