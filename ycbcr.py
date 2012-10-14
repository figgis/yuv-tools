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

        self.__check()
        self.__calc()
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

        value_frames = []

        with open(self.filename, 'rb') as fd_1, \
             open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                frame1 = array.array('B')
                frame2 = array.array('B')

                frame1.fromfile(fd_1, self.frame_size_in)
                frame2.fromfile(fd_2, self.frame_size_in)

                frame_mse = sum_square_err(frame1, frame2) / float(len(frame1))
                frame_psnr = psnr(frame_mse)
                value_frames.append(frame_psnr)

                print "frame: %-10s %.4f" % (i, frame_psnr)

        return value_frames
#-------------------------------------------------------------------------------
    def __check(self):
        """
        Basic consistency checks
        TODO: add...
        """
        pass
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
    def split(self):
        """
        Split a file into separate frames.
        """
        src_yuv = open(self.filename, 'rb')

        filecnt = 0
        while True:
            data = src_yuv.read(self.frame_size_in)
            if data:
                fname = "frame" + "%s" % filecnt + ".yuv"
                dst_yuv = open(fname, 'wb')
                dst_yuv.write(data)           # write read data into new file
                print "writing frame", filecnt
                dst_yuv.close()
                filecnt += 1
            else:
                break
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
    yuv.psnr()

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
            choices=['IYUV', 'UYVY', 'YV12', 'YVYU'],
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
            choices=['IYUV', 'UYVY', 'YV12', 'YVYU', '422'],
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
            help='Calculate PSNR for each frame and color-plane',
            parents=[parent_parser])
    parser_psnr.add_argument('filename_diff', type=str, help='filename')
    parser_psnr.set_defaults(func=__cmd_psnr)

    # let parse_args() do the job of calling the appropriate function
    # after argument parsing is complete
    args = parser.parse_args()
    t1 = time.clock()
    args.func(args)
    t2 = time.clock()
    print "\nTime: ", round(t2-t1, 4)
