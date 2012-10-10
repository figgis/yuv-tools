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

#import cProfile

#-------------------------------------------------------------------------------
class YCbCr:
    """
    Tools to work with raw video in YCbCr format.

    For description of the supported formats, see

        http://www.fourcc.org/yuv.php

    YUV video sequences can be downloaded from

        http://trace.eas.asu.edu/yuv/

    Supports the following YCbCr-formats:

        {IYUV, UYVY, YV12, YVYU}

    Main reason for this is that those are the formats supported by

        http://www.libsdl.org/
        http://www.libsdl.org/docs/html/sdloverlay.html
    """
    def __init__(self, width=0, height=0, filename=None, yuv_format_in=None,
            yuv_format_out=None, filename_out=None, filename_diff=None,
            func=None):
        #TODO: fix
        self.sampling = {
                'IYUV':1.5,
                'UYVY':2,
                'YV12':1.5,
                'YVYU':2,
                '422' :2,
                }

        self.filename = filename
        self.filename_out = filename_out
        self.filename_diff = filename_diff
        self.width = width
        self.height = height
        self.yuv_format_in = yuv_format_in
        self.yuv_format_out = yuv_format_out
        self.file_size_in_bytes = os.stat(self.filename)[6]
        self.frame_size_in = int(self.width * self.height *
                              self.sampling[self.yuv_format_in])
        if yuv_format_out:
            self.frame_size_out = int(self.width * self.height *
                    self.sampling[self.yuv_format_out])
        else:
            self.frame_size_out = None

        self.num_frames = self.file_size_in_bytes / self.frame_size_in

        self.reader = {
            'YV12': self._read_yv12,
            'IYUV': self._read_iyuv,
            'UYVY': self._read_uyvy,
            'YVYU': self._read_yvyu,
        }

        self.writer = {
            'YV12': self._write_yv12,
            'IYUV': self._write_iyuv,
            'UYVY': self._write_uyvy,
            'YVYU': self._write_yvyu,
            '422' : self._write_422,
        }

        self._check()
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
        print "Filesize (bytes):", self.file_size_in_bytes
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
        with open(self.filename, 'rb') as fd_in, open(self.filename_out, 'wb') as fd_out:
            for i in xrange(self.num_frames):
                # 1. read one frame, result in self.{y, cb, cr}
                self.reader[self.yuv_format_in](fd_in)
                # 2. converts one frame self.{y,cb, cr} to correct format and
                #    writes it to file
                self.writer[self.yuv_format_out](fd_out)
                sys.stdout.write('.')
                sys.stdout.flush()
#-------------------------------------------------------------------------------
    def diff(self):
        """
        Produces a YV12 file containing the luma-difference between
        two files.
        """
        out = self.filename.split('.')[0] + '_' + \
              self.filename_diff.split('.')[0] + '_diff.yuv'

        fd_out = open(out, 'wb')
        with open(self.filename, 'rb') as fd_1, open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                self.reader[self.yuv_format_in](fd_1)
                data1 = [i for i in self.y]
                self.reader[self.yuv_format_in](fd_2)
                data2 = [i for i in self.y]

                D = []
                for x, y in zip(data1, data2):
                    D.append(self._clip(0x80-abs(x-y)))

                fd_out.write(array.array('B', D).tostring())

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

        with open(self.filename, 'rb') as fd_1, open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                frame1 = array.array('B')
                frame2 = array.array('B')

                frame1.fromfile(fd_1, self.frame_size_in)
                frame2.fromfile(fd_2, self.frame_size_in)

                frame_mse = sum_square_err(frame1, frame2) / float(len(frame1))
                frame_psnr = psnr(frame_mse)

                print "frame: %-10s %.4f" % (i, frame_psnr)
#-------------------------------------------------------------------------------
    def _check(self):
        """
        Basic consistency checks
        """

#-------------------------------------------------------------------------------
    def _read_yv12(self, fd):
        """
        Read one frame
        """
        self.y = array.array('B')
        self.cb = array.array('B')
        self.cr = array.array('B')

        self.y.fromfile(fd, self.width * self.height)
        self.cb.fromfile(fd, self.width * self.height / 4)
        self.cr.fromfile(fd, self.width * self.height / 4)
#-------------------------------------------------------------------------------
    def _read_uyvy(self, fd):
        """
        Read one frame
        """
        self.y = array.array('B')
        self.cb = array.array('B')
        self.cr = array.array('B')

        raw = array.array('B')
        raw.fromfile(fd, self.frame_size_in)

        self.y = raw[1::2]     # y
        self.cb = raw[0::4]    # u
        self.cr = raw[2::4]    # v
#-------------------------------------------------------------------------------
    def _read_yvyu(self, fd):
        """
        Read one frame
        """
        self.y = array.array('B')
        self.cb = array.array('B')
        self.cr = array.array('B')

        raw = array.array('B')
        raw.fromfile(fd, self.frame_size_in)

        # Y0|V0|Y1|U0
        self.y = raw[0::2]     # y
        self.cr = raw[1::4]    # v
        self.cb = raw[3::4]    # u
#-------------------------------------------------------------------------------
    def _read_iyuv(self, fd):
        """
        read one frame
        IYUV is YV12 where the croma-planes have switched places.
        """
        self.y = array.array('B')
        self.cb = array.array('B')
        self.cr = array.array('B')

        self.y.fromfile(fd, self.width * self.height)
        self.cr.fromfile(fd, self.width * self.height / 4)
        self.cb.fromfile(fd, self.width * self.height / 4)
#-------------------------------------------------------------------------------
    def _write_yvyu(self, fd):
        """
        write one frame
        handle re-sampling of frame from 4:2:0 -> 4:2:2
        """
        if self.yuv_format_in in ['YV12', 'IYUV']:
            cb = [0] * (self.width / 2 * self.height)
            cr = [0] * (self.width / 2 * self.height)

            self.cb = self._conv420to422(self.cb, cb)
            self.cr = self._conv420to422(self.cr, cr)

        yvyu = [0] * self.frame_size_out
        yvyu[0::2] = self.y
        yvyu[1::4] = self.cr
        yvyu[3::4] = self.cb

        fd.write(array.array('B', yvyu).tostring())
#-------------------------------------------------------------------------------
    def _write_yv12(self, fd):
        """
        write one frame
        handle re-sampling of frame from 4:2:2 -> 4:2:0
        """
        if self.yuv_format_in in ['UYVY', 'YVYU']:
            cb = [0] * (self.width  * self.height / 4)
            cr = [0] * (self.width  * self.height / 4)

            self.cb = self._conv422to420(self.cb, cb)
            self.cr = self._conv422to420(self.cr, cr)

        fd.write(array.array('B', self.y).tostring())
        fd.write(array.array('B', self.cb).tostring())
        fd.write(array.array('B', self.cr).tostring())
#-------------------------------------------------------------------------------
    def _write_uyvy(self, fd):
        """
        write one frame
        handle re-sampling of frame from 4:2:0 -> 4:2:2
        """
        if self.yuv_format_in in ['YV12', 'IYUV']:
            cb = [0] * (self.width / 2 * self.height)
            cr = [0] * (self.width / 2 * self.height)

            self.cb = self._conv420to422(self.cb, cb)
            self.cr = self._conv420to422(self.cr, cr)

        uyvy = [0] * self.frame_size_out
        uyvy[1::2] = self.y
        uyvy[0::4] = self.cb
        uyvy[2::4] = self.cr

        fd.write(array.array('B', uyvy).tostring())
#-------------------------------------------------------------------------------
    def _write_iyuv(self, fd):
        """
        write one frame
        handle re-sampling of frame from 4:2:2 -> 4:2:0
        """
        if self.yuv_format_in in ['UYVY', 'YVYU']:
            cb = [0] * (self.width * self.height / 4)
            cr = [0] * (self.width * self.height / 4)

            self.cb = self._conv422to420(self.cb, cb)
            self.cr = self._conv422to420(self.cr, cr)

        fd.write(array.array('B', self.y).tostring())
        fd.write(array.array('B', self.cr).tostring())
        fd.write(array.array('B', self.cb).tostring())
#-------------------------------------------------------------------------------
    def _write_422(self, fd):
        """
        write one frame as plane-separated.
        handle re-sampling of frame from 4:2:0 -> 4:2:2
        """
        if self.yuv_format_in in ['YV12', 'IYUV']:
            cb = [0] * (self.width * self.height / 2)
            cr = [0] * (self.width * self.height / 2)

            self.cb = self._conv420to422(self.cb, cb)
            self.cr = self._conv420to422(self.cr, cr)

        fd.write(array.array('B', self.y).tostring())
        fd.write(array.array('B', self.cb).tostring())
        fd.write(array.array('B', self.cr).tostring())
#-------------------------------------------------------------------------------
    def _conv420to422(self, src, dst):
        """
        420 to 422 - vertical 1:2 interpolation filter

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

                a = (3*src[i+w*jm3]
                   -16*src[i+w*jm2]
                   +67*src[i+w*jm1]
                  +227*src[i+w*j]
                   -32*src[i+w*jp1]
                    +7*src[i+w*jp2]+128)>>8

                dst[i+w*j2] = a if a > 0 else 0
                dst[i+w*j2] = a if a < 255 else 255

                b = (3*src[i+w*jp3]
                   -16*src[i+w*jp2]
                   +67*src[i+w*jp1]
                  +227*src[i+w*j]
                   -32*src[i+w*jm1]
                   +7*src[i+w*jm2]+128)>>8

                dst[i+w*(j2+1)] = b if b > 0 else 0
                dst[i+w*(j2+1)] = b if b < 255 else 255
        return dst
#-------------------------------------------------------------------------------
    def _conv422to420(self, src, dst):
        """
        422 -> 420

        http://www.mpeg.org/MPEG/video/mssg-free-mpeg-software.html
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

                # FIR filter with 0.5 sample interval phase shift
                a = ( 228*(src[i+w*j]  +src[i+w*jp1])
                      +70*(src[i+w*jm1]+src[i+w*jp2])
                      -37*(src[i+w*jm2]+src[i+w*jp3])
                      -21*(src[i+w*jm3]+src[i+w*jp4])
                      +11*(src[i+w*jm4]+src[i+w*jp5])
                      +5*(src[i+w*jm5]+src[i+w*jp6])+256)>>9

                dst[i+w*(j>>1)] = a if a > 0 else 0
                dst[i+w*(j>>1)] = a if a < 255 else 255
        return dst
#-------------------------------------------------------------------------------
    def _g3(self, string):
        """
        return a array of int from a string
        """
        pass
        #return [int(binascii.hexlify(i), 16) for i in string]
#        return array.array('B', string).tolist()
        #return map(ord, string)
#-------------------------------------------------------------------------------
    def _g1(self, my_list):
        """create a string of integer ASCII values from a list of int"""
        pass
#        return array.array('B', my_list).tostring()
#-------------------------------------------------------------------------------
    def _clip(self, data):
        """
        clip function
        """
        if data > 255:
            data = 255
        elif data < 0:
            data = 0
        return data
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

def _cmd_info(args):
    YCbCr(**vars(args)).show()

def _cmd_split(args):
    Y = YCbCr(**vars(args))
    Y.show()
    Y.split()

def _cmd_convert(args):
    Y = YCbCr(**vars(args))
    Y.show()
    Y.convert()

def _cmd_diff(args):
    Y = YCbCr(**vars(args))
    Y.show()
    Y.diff()

def _cmd_psnr(args):
    Y = YCbCr(**vars(args))
    Y.psnr()

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
    parser_info.set_defaults(func=_cmd_info)

    # create parser for the 'split' command
    parser_split = subparsers.add_parser('split',
            help='Split a YCbCr file into individual frames',
            parents=[parent_parser])
    parser_split.set_defaults(func=_cmd_split)

    # create parser for the 'convert' command
    parser_convert = subparsers.add_parser('convert',
            help='YCbCr format conversion',
            parents=[parent_parser])
    parser_convert.add_argument('yuv_format_out', type=str,
            choices=['IYUV', 'UYVY', 'YV12', 'YVYU', '422'],
            help='valid output-formats')
    parser_convert.add_argument('filename_out', type=str,
            help='file to write to')
    parser_convert.set_defaults(func=_cmd_convert)

    # create parser for the 'diff' command
    parser_diff = subparsers.add_parser('diff',
            help='Create diff between two YCbCr files',
            parents=[parent_parser])
    parser_diff.add_argument('filename_diff', type=str, help='filename')
    parser_diff.set_defaults(func=_cmd_diff)

    # create parser for the 'psnr' command
    parser_psnr = subparsers.add_parser('psnr',
            help='Calculate PSNR for each frame and color-plane',
            parents=[parent_parser])
    parser_psnr.add_argument('filename_diff', type=str, help='filename')
    parser_psnr.set_defaults(func=_cmd_psnr)

    # let parse_args() do the job of calling the appropriate function
    # after argument parsing is complete
    args = parser.parse_args()
    t1 = time.clock()
    args.func(args)
    t2 = time.clock()
    print "\nTime: ", round(t2-t1, 4)
