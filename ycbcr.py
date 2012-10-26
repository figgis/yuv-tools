#!/usr/bin/env python

"""
Tools for working with YCbCr data.
"""

import argparse
import array
import time
import math
import sys
import os

class Format(Exception):
    pass

#-------------------------------------------------------------------------------
class YCbCr:
    """
    Tools to work with raw video in YCbCr format.

    For description of the supported formats, see

        http://www.fourcc.org/yuv.php

    YUV video sequences can be downloaded from

        http://trace.eas.asu.edu/yuv/

    Supports the following YCbCr-formats:

        {IYUV, UYVY, YV12, YVYU, YUY2}

    Main reason for this is that those are the formats supported by

        http://www.libsdl.org/
        http://www.libsdl.org/docs/html/sdloverlay.html
    """
    def __init__(self, width=0, height=0, filename=None, yuv_format_in=None,
            yuv_format_out=None, filename_out=None, filename_diff=None,
            func=None):

        if yuv_format_in not in ['IYUV', 'UYVY', 'YV12', 'YVYU', 'YUY2']:
            raise NameError('format not supported! "%s"' % yuv_format_in)
        if yuv_format_out not in ['IYUV', 'UYVY', 'YV12', 'YVYU', 'YUY2', '422', None]:
            raise NameError('format not supported! "%s"' % yuv_format_out)

        self.filename = filename
        self.filename_out = filename_out
        self.filename_diff = filename_diff
        self.width = width
        self.height = height
        self.yuv_format_in = yuv_format_in
        self.yuv_format_out = yuv_format_out

        self.frame_size_in = 0
        self.frame_size_out = None
        self.num_frames = 0

        self.y = None
        self.cb = None
        self.cr = None

        self.layout = None

        self.__calc()
        self.__check()
#-------------------------------------------------------------------------------
    def show(self):
        """
        Display basic info.
        """
        print
        print "Filename (in):", self.filename
        print "Filename (out):", self.filename_out
        print "Format (in):", self.yuv_format_in
        print "Format (out):", self.yuv_format_out
        print "Width:", self.width
        print "Height:", self.height
        print "Filesize (bytes):", os.stat(self.filename)[6]
        print "Num frames:", self.num_frames
        print "Size of 1 frame (in) (bytes):", self.frame_size_in
        print "Size of 1 frame (out) (bytes):", self.frame_size_out
        print
#-------------------------------------------------------------------------------
    def convert(self):
        """
        Format-conversion between the supported formats.
        4:2:0 to 4:2:2 interpolation and 4:2:2 to 4:2:0
        subsampling when necessary.
        """
        with open(self.filename, 'rb') as fd_in, \
             open(self.filename_out, 'wb') as fd_out:
            for i in xrange(self.num_frames):
                # 1. read one frame, result in self.{y, cb, cr}
                self.__read_frame(fd_in)
                # 2. converts one frame self.{y,cb, cr} to correct format and
                #    write it to file
                self.__write_frame(fd_out)
                sys.stdout.write('.')
                sys.stdout.flush()
#-------------------------------------------------------------------------------
    def diff(self):
        """
        Produces a YV12 file containing the luma-difference between
        two files.
        """
        def clip(data):
            """
            clip function
            """
            if data > 255:
                data = 255
            elif data < 0:
                data = 0
            return data

        base1 = os.path.basename(self.filename)
        base2 = os.path.basename(self.filename_diff)
        out = os.path.splitext(base1)[0] +'_' + \
              os.path.splitext(base2)[0] + '_diff.yuv'

        chroma = [0x80] * (self.width*self.height/2)
        fd_out = open(out, 'wb')
        with open(self.filename, 'rb') as fd_1, \
             open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_1)
                data1 = list(self.y)
                self.__read_frame(fd_2)
                data2 = list(self.y)

                D = []
                for x, y in zip(data1, data2):
                    D.append(clip(0x80-abs(x-y)))

                fd_out.write(array.array('B', D).tostring())
                fd_out.write(array.array('B', chroma).tostring())

                sys.stdout.write('.')
                sys.stdout.flush()
        fd_out.close()
#-------------------------------------------------------------------------------
    def psnr(self):
        """
        Well, PSNR calculations on a frame-basis between two files.

        http://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        """
        def psnr(mse):
            log10 = math.log10
            if mse == 0:
                return float("nan")
            return 10.0*log10(float(256*256)/float(mse))

        def sum_square_err(data1, data2):
            return sum((a-b)*(a-b) for a, b in zip(data1, data2))

        with open(self.filename, 'rb') as fd_1, \
             open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_1)
                y1 = list(self.y)
                cb1 = list(self.cb)
                cr1 = list(self.cr)
                self.__read_frame(fd_2)
                y2 = list(self.y)
                cb2 = list(self.cb)
                cr2 = list(self.cr)

                frame1 = [y1, cb1, cr1]
                frame2 = [y2, cb2, cr2]

                mse = [sum_square_err(x, y) / float(len(x)) for x, y in zip(frame1, frame2)]
                yield [psnr(i) for i in mse]
#-------------------------------------------------------------------------------
    def ssim(self):
        """
        http://en.wikipedia.org/wiki/Structural_similarity

        implementation using scipy and numpy from
        http://isit.u-clermont1.fr/~anvacava/code.html
        """
        import numpy
        import scipy.ndimage
        from numpy.ma.core import exp
        from scipy.constants.constants import pi

        def compute_ssim(img_mat_1, img_mat_2):
            #Variables for Gaussian kernel definition
            gaussian_kernel_sigma=1.5
            gaussian_kernel_width=11
            gaussian_kernel=numpy.zeros((gaussian_kernel_width,gaussian_kernel_width))

            #Fill Gaussian kernel
            for i in range(gaussian_kernel_width):
                for j in range(gaussian_kernel_width):
                    gaussian_kernel[i,j]=\
                    (1/(2*pi*(gaussian_kernel_sigma**2)))*\
                    exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

            #Convert image matrices to double precision (like in the Matlab version)
            img_mat_1=img_mat_1.astype(numpy.float)
            img_mat_2=img_mat_2.astype(numpy.float)

            #Squares of input matrices
            img_mat_1_sq=img_mat_1**2
            img_mat_2_sq=img_mat_2**2
            img_mat_12=img_mat_1*img_mat_2

            #Means obtained by Gaussian filtering of inputs
            img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
            img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)

            #Squares of means
            img_mat_mu_1_sq=img_mat_mu_1**2
            img_mat_mu_2_sq=img_mat_mu_2**2
            img_mat_mu_12=img_mat_mu_1*img_mat_mu_2

            #Variances obtained by Gaussian filtering of inputs' squares
            img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
            img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)

            #Covariance
            img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)

            #Centered squares of variances
            img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
            img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
            img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;

            #c1/c2 constants
            #First use: manual fitting
            c_1=6.5025
            c_2=58.5225

            #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
            l=255
            k_1=0.01
            c_1=(k_1*l)**2
            k_2=0.03
            c_2=(k_2*l)**2

            #Numerator of SSIM
            num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
            #Denominator of SSIM
            den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
            (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
            #SSIM
            ssim_map=num_ssim/den_ssim
            index=numpy.average(ssim_map)

            return index

        def l2n(x, w, h):
            """
            list 2 numpy, including reshape
            """
            n = numpy.array(x, dtype=numpy.uint8)
            return numpy.reshape(n, (h, w))


        with open(self.filename, 'rb') as fd_1, \
             open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_1)
                data1 = list(self.y)
                self.__read_frame(fd_2)
                data2 = list(self.y)

                yield compute_ssim(l2n(data1, self.width, self.height),
                        l2n(data2, self.width, self.height))
#-------------------------------------------------------------------------------
    def get_luma(self, alt_fname=False):
        """
        Generator to get luminance-data for all frames
        """
        if alt_fname:
            fname = alt_fname
        else:
            fname = self.filename

        with open(fname, 'rb') as fd_in:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_in)
                yield self.y
#-------------------------------------------------------------------------------
    def __check(self):
        """
        Basic consistency checks to prevent fumbly-fingers
        - width & height even multiples of 16
        - number of frames divides file-size evenly
        - for diff-cmd, file-sizes match
        """
        if self.width & 0xF != 0:
            print >> sys.stderr, "[WARNING] - width not divisable by 16"
        if self.height & 0xF != 0:
            print >> sys.stderr, "[WARNING] - hight not divisable by 16"

        size = os.path.getsize(self.filename)
        if not self.num_frames == size / float(self.frame_size_in):
            print >> sys.stderr, "[WARNING] - # frames not integer"

        if self.filename_diff:
            if not os.path.getsize(self.filename) == os.path.getsize(self.filename_diff):
                print >> sys.stderr, "[WARNING] - file-sizes are not equal"
#-------------------------------------------------------------------------------
    def __calc(self):
        """
        Setup some variables and calculate the layout-dictionary
        containing info on howto read/write the various supported
        formats
        """
        sampling = { 'IYUV':1.5,
                     'UYVY':2,
                     'YV12':1.5,
                     'YVYU':2,
                     '422' :2,
                     'YUY2':2}

        self.frame_size_in = int(self.width * self.height *
                              sampling[self.yuv_format_in])
        if self.yuv_format_out:
            self.frame_size_out = int(self.width * self.height *
                    sampling[self.yuv_format_out])

        self.num_frames = os.stat(self.filename)[6] / self.frame_size_in

        # helper variables for indexing planar-formats
        # (start, end), a bit ugly!
        ys = 0
        ye = self.width * self.height
        cbs = ye
        cbe = cbs+ye/4
        cbee = cbs+ye/2
        crs = cbe
        crss = cbee
        cre = crs+ye/4
        cree = crss+ye/2

        # Instead of using separate read/write function for each supported
        # format, store start_pos and stride for each format in a dictionary
        # and use extended indexing to place the data where it belongs.
        # Works great for packed formats; for planar use normal slicing
        self.layout = {
                'UYVY': { # U0|Y0|V0|Y1
                    'y_start_pos':  1, 'y_stride'  :   2,
                    'cb_start_pos': 0, 'cb_stride':    4,
                    'cr_start_pos': 2, 'cr_stride':    4,
                    },
                'YVYU': { # Y0|V0|Y1|U0
                    'y_start_pos':  0, 'y_stride':     2,
                    'cb_start_pos': 3, 'cb_stride':    4,
                    'cr_start_pos': 1, 'cr_stride':    4,
                    },
                'YUY2': { # Y0|U0|Y1|V0
                    'y_start_pos':  0, 'y_stride':     2,
                    'cb_start_pos': 1, 'cb_stride':    4,
                    'cr_start_pos': 3, 'cr_stride':    4,
                    },
                'YV12': { # Y|U|V
                    'y_start_pos':  ys,  'y_stride':   ye,
                    'cb_start_pos': cbs, 'cb_stride':  cbe,
                    'cr_start_pos': crs, 'cr_stride':  cre,
                    },
                'IYUV': { # Y|V|U
                    'y_start_pos':  ys,  'y_stride':   ye,
                    'cb_start_pos': crs, 'cb_stride':  cre,
                    'cr_start_pos': cbs, 'cr_stride':  cbe,
                    },
                '422': { # Y|U|V
                    'y_start_pos':  ys,   'y_stride':   ye,
                    'cb_start_pos': cbs,  'cb_stride':  cbee,
                    'cr_start_pos': crss, 'cr_stride':  cree,
                    }
                }
#-------------------------------------------------------------------------------
    def __read_frame(self, fd):
        """
        Use extended indexing to read 1 frame into self.{y, cb, cr}
        """
        packed = ('UYVY', 'YVYU','YUY2')

        self.y = array.array('B')
        self.cb = array.array('B')
        self.cr = array.array('B')

        raw = array.array('B')
        raw.fromfile(fd, self.frame_size_in)

        if self.yuv_format_in in packed:
            self.y  = raw[self.layout[self.yuv_format_in]['y_start_pos']::
                          self.layout[self.yuv_format_in]['y_stride']]
            self.cb = raw[self.layout[self.yuv_format_in]['cb_start_pos']::
                          self.layout[self.yuv_format_in]['cb_stride']]
            self.cr = raw[self.layout[self.yuv_format_in]['cr_start_pos']::
                          self.layout[self.yuv_format_in]['cb_stride']]
        else:
            self.y  = raw[self.layout[self.yuv_format_in]['y_start_pos']:
                          self.layout[self.yuv_format_in]['y_stride']]
            self.cb = raw[self.layout[self.yuv_format_in]['cb_start_pos']:
                          self.layout[self.yuv_format_in]['cb_stride']]
            self.cr = raw[self.layout[self.yuv_format_in]['cr_start_pos']:
                          self.layout[self.yuv_format_in]['cr_stride']]
#-------------------------------------------------------------------------------
    def __write_frame(self, fd):
        """
        Use extended indexing to write 1 frame, including re-sampling and
        format conversion
        """
        packed = ('UYVY', 'YVYU','YUY2')

        self.__resample()
        data = [0] * self.frame_size_out

        if self.yuv_format_out in packed:
            data[self.layout[self.yuv_format_out]['y_start_pos']::
                 self.layout[self.yuv_format_out]['y_stride']] = self.y
            data[self.layout[self.yuv_format_out]['cb_start_pos']::
                 self.layout[self.yuv_format_out]['cb_stride']] = self.cb
            data[self.layout[self.yuv_format_out]['cr_start_pos']::
                 self.layout[self.yuv_format_out]['cr_stride']] = self.cr
        else:
            data[self.layout[self.yuv_format_out]['y_start_pos']:
                 self.layout[self.yuv_format_out]['y_stride']] = self.y
            data[self.layout[self.yuv_format_out]['cb_start_pos']:
                 self.layout[self.yuv_format_out]['cb_stride']] = self.cb
            data[self.layout[self.yuv_format_out]['cr_start_pos']:
                 self.layout[self.yuv_format_out]['cr_stride']] = self.cr

        fd.write(array.array('B', data).tostring())
#-------------------------------------------------------------------------------
    def __resample(self):
        """
        Handle 420 -> 422 and 422 -> 420
        """
        f420 = ('YV12', 'IYUV')
        f422 = ('UYVY', 'YVYU', 'YUY2', '422')

        if self.yuv_format_in in f420 and self.yuv_format_out in f422:
            cb = [0] * (self.width * self.height / 2)
            cr = [0] * (self.width * self.height / 2)

            self.cb = self.__conv420to422(self.cb, cb)
            self.cr = self.__conv420to422(self.cr, cr)

        if self.yuv_format_in in f422 and self.yuv_format_out in f420:
            cb = [0] * (self.width * self.height / 4)
            cr = [0] * (self.width * self.height / 4)

            self.cb = self.__conv422to420(self.cb, cb)
            self.cr = self.__conv422to420(self.cr, cr)
#-------------------------------------------------------------------------------
    def __conv420to422(self, src, dst):
        """
        420 to 422 - vertical 1:2 interpolation filter

        Bit-exact with
        http://www.mpeg.org/MPEG/video/mssg-free-mpeg-software.html
        """
        w = self.width >> 1
        h = self.height >> 1

        for i in xrange(w):
            for j in xrange(h):
                j2 = j << 1
                jm3 = 0 if (j<3) else j-3
                jm2 = 0 if (j<2) else j-2
                jm1 = 0 if (j<1) else j-1
                jp1 = j+1 if (j<h-1) else h-1
                jp2 = j+2 if (j<h-2) else h-1
                jp3 = j+3 if (j<h-3) else h-1

                pel = (3*src[i+w*jm3]
                     -16*src[i+w*jm2]
                     +67*src[i+w*jm1]
                    +227*src[i+w*j]
                     -32*src[i+w*jp1]
                      +7*src[i+w*jp2]+128)>>8

                dst[i+w*j2] = pel if pel > 0 else 0
                dst[i+w*j2] = pel if pel < 255 else 255

                pel = (3*src[i+w*jp3]
                     -16*src[i+w*jp2]
                     +67*src[i+w*jp1]
                    +227*src[i+w*j]
                     -32*src[i+w*jm1]
                     +7*src[i+w*jm2]+128)>>8

                dst[i+w*(j2+1)] = pel if pel > 0 else 0
                dst[i+w*(j2+1)] = pel if pel < 255 else 255
        return dst
#-------------------------------------------------------------------------------
    def __conv422to420(self, src, dst):
        """
        422 -> 420

        http://www.mpeg.org/MPEG/video/mssg-free-mpeg-software.html
        although reference implementation reads data out-of-bounds,
        jp6 is the offending parameter. linking with electric-fence
        core-dumps. Bit-excact after change.
        """
        w = self.width >> 1
        h = self.height

        for i in xrange(w):
            for j in xrange(0, h, 2):
                jm5 = 0 if (j<5) else j-5
                jm4 = 0 if (j<4) else j-4
                jm3 = 0 if (j<3) else j-3
                jm2 = 0 if (j<2) else j-2
                jm1 = 0 if (j<1) else j-1
                jp1 = j+1 if (j<h-1) else h-1
                jp2 = j+2 if (j<h-2) else h-1
                jp3 = j+3 if (j<h-3) else h-1
                jp4 = j+4 if (j<h-4) else h-1
                jp5 = j+5 if (j<h-5) else h-1
                jp6 = j+5 if (j<h-5) else h-1 # something strange here
                                              # changed j+6 into j+5

                # FIR filter with 0.5 sample interval phase shift
                pel = ( 228*(src[i+w*j]  +src[i+w*jp1])
                      +70*(src[i+w*jm1]+src[i+w*jp2])
                      -37*(src[i+w*jm2]+src[i+w*jp3])
                      -21*(src[i+w*jm3]+src[i+w*jp4])
                      +11*(src[i+w*jm4]+src[i+w*jp5])
                      +5*(src[i+w*jm5]+src[i+w*jp6])+256)>>9

                dst[i+w*(j>>1)] = pel if pel > 0 else 0
                dst[i+w*(j>>1)] = pel if pel < 255 else 255
        return dst
#-------------------------------------------------------------------------------
    def __rgb2ycbcr(self, r, g, b):
        """
        (r,g,b) -> (y, cb, cr)

        Conversion to YCbCr color space.
        CCIR 601 formulas from "Digital Pictures by Natravali and Haskell, page 120.
        """
        y  =  _clip2UInt8( 0.257*r + 0.504*g + 0.098*b + 16)
        cb =  _clip2UInt8(-0.148*r - 0.291*g + 0.439*b + 128)
        cr =  _clip2UInt8( 0.439*r - 0.368*g - 0.071*b + 128)

        return (y, cb, cr)
#-------------------------------------------------------------------------------
    def __ycbcr2rgb(self, y, cb, cr):
        """
        (y,cb,cr) -> (r, g, b)

        Conversion to RGB color space.
        CCIR 601 formulas from "Digital Pictures by Natravali and Haskell, page 120.
        """
        y =  y-16
        cb = cb-128
        cr = cr-128

        r = _clip2UInt8(1.164*y            + 1.596*cr)
        g = _clip2UInt8(1.164*y - 0.392*cb - 0.813*cr)
        b = _clip2UInt8(1.164*y + 2.017*cb           )

        return (r, g, b)
#-------------------------------------------------------------------------------
    def _clip2UInt8(self, d):
        "Clip d to interval 0-255"

        if (d < 0 ):
            return 0

        if (d > 255):
            return 255

        return int(round(d))
#-------------------------------------------------------------------------------
    def split(self):
        """
        Split a file into separate frames.
        """
        src_yuv = open(self.filename, 'rb')

        for i in xrange(self.num_frames):
            data = src_yuv.read(self.frame_size_in)
            fname = "frame" + "%d" % i + ".yuv"
            dst_yuv = open(fname, 'wb')
            dst_yuv.write(data)           # write read data into new file
            dst_yuv.close()
        src_yuv.close()
#-------------------------------------------------------------------------------

# Helper functions

def __cmd_info(arg):
    YCbCr(**vars(arg)).show()

def __cmd_split(arg):
    yuv = YCbCr(**vars(arg))
    yuv.show()
    yuv.split()

def __cmd_convert(arg):
    yuv = YCbCr(**vars(arg))
    yuv.show()
    yuv.convert()

def __cmd_diff(arg):
    yuv = YCbCr(**vars(arg))
    yuv.show()
    yuv.diff()

def __cmd_psnr(arg):
    yuv = YCbCr(**vars(arg))
    for i, n in enumerate(yuv.psnr()):
        print i, n

def __cmd_ssim(arg):
    yuv = YCbCr(**vars(arg))
    for i, n in enumerate(yuv.ssim()):
        print i, n

def __cmd_get_luma(arg):
    yuv = YCbCr(**vars(arg))
    return yuv.get_luma()

if __name__ == '__main__':
    # create the top-level parser
    parser = argparse.ArgumentParser(description='YCbCr tools',
            epilog=' Be careful with those bits')
    subparsers = parser.add_subparsers(title='subcommands',
                                       help='additional help')

    # parent, common arguments for functions
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('filename', type=str, help='filename')
    parent_parser.add_argument('width', type=int)
    parent_parser.add_argument('height', type=int)
    parent_parser.add_argument('yuv_format_in', type=str,
            choices=['IYUV', 'UYVY', 'YV12', 'YVYU', 'YUY2'],
            help='valid input-formats')

    # create parser for the 'info' command
    parser_info = subparsers.add_parser('info',
            help='Basic info about YCbCr file',
            parents=[parent_parser])
    parser_info.set_defaults(func=__cmd_info)

    # create parser for the 'split' command
    parser_split = subparsers.add_parser('split',
            help='Split a YCbCr file into individual frames',
            parents=[parent_parser])
    parser_split.set_defaults(func=__cmd_split)

    # create parser for the 'convert' command
    parser_convert = subparsers.add_parser('convert',
            help='YCbCr format conversion',
            parents=[parent_parser])
    parser_convert.add_argument('yuv_format_out', type=str,
            choices=['IYUV', 'UYVY', 'YV12', 'YVYU', '422', 'YUY2'],
            help='valid output-formats')
    parser_convert.add_argument('filename_out', type=str,
            help='file to write to')
    parser_convert.set_defaults(func=__cmd_convert)

    # create parser for the 'diff' command
    parser_diff = subparsers.add_parser('diff',
            help='Create diff between two YCbCr files',
            parents=[parent_parser])
    parser_diff.add_argument('filename_diff', type=str, help='filename')
    parser_diff.set_defaults(func=__cmd_diff)

    # create parser for the 'psnr' command
    parser_psnr = subparsers.add_parser('psnr',
            help='Calculate PSNR for each frame, luma data only',
            parents=[parent_parser])
    parser_psnr.add_argument('filename_diff', type=str, help='filename')
    parser_psnr.set_defaults(func=__cmd_psnr)

    # create parser for the 'ssim' command
    parser_psnr = subparsers.add_parser('ssim',
            help='Calculate ssim for each frame, luma data only',
            parents=[parent_parser])
    parser_psnr.add_argument('filename_diff', type=str, help='filename')
    parser_psnr.set_defaults(func=__cmd_ssim)

    # create parser for the 'get_luma' command
    parser_info = subparsers.add_parser('get_luma',
            help='Return luminance-data for each frame. Generator',
            parents=[parent_parser])
    parser_info.set_defaults(func=__cmd_get_luma)

    # let parse_args() do the job of calling the appropriate function
    # after argument parsing is complete
    args = parser.parse_args()
    t1 = time.clock()
    args.func(args)
    t2 = time.clock()
    print "\nTime: ", round(t2-t1, 4)
