#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Tool-set to work with raw video in YCbCr format.

For description of the supported formats, see

    http://www.fourcc.org/yuv.php

YUV video sequences can be downloaded from

    http://trace.eas.asu.edu/yuv/
"""

import argparse
import binascii
import array
import sys
import os

#-------------------------------------------------------------------------------
class YCbCr:
    def __init__(self, width=0, height=0, filename=None, yuv_format_in=None,
            yuv_format_out=None, filename_out=None, filename_diff=None, func=None):
        #TODO: fix
        self.sampling = {
                'IYUV':1.5,
                'UYVY':2,
                'YV12':1.5,
                'YVYU':2
                }

        self.supported_formats = ['IYUV', 'UYVY', 'YV12', 'YVYU']
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
        self.num_frames =self.file_size_in_bytes / self.frame_size_in

        if yuv_format_out:
            self.zero = [0 for i in xrange(self.frame_size_out)]

        # Store each plane as a separate buffer
        self.y = []
        self.cb = []
        self.cr = []

        self.reader = {
            'YV12': self._read_YV12,
            'IYUV': self._read_IYUV,
            'UYVY': self._read_UYVY,
            'YVYU': self._read_YVYU,
        }

        self.writer = {
            'YV12': self._write_YV12,
            'IYUV': self._write_IYUV,
            'UYVY': self._write_UYVY,
            'YVYU': self._write_YVYU,
        }
#-------------------------------------------------------------------------------
    def show(self):
        """
        show stuff
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
        Main-loop for yuv-format-conversion
        """
        with open(self.filename, 'rb') as fd_in, open(self.filename_out, 'wb') as fd_out:
            for i in xrange(self.num_frames):
                # 1. read one frame, read() places the result in self.{y, cb, cr}
                self.reader[self.yuv_format_in](fd_in)
                # 2. write() converts one frame self.{y,cb, cr} to correct format and
                #    writes it to file
                self.writer[self.yuv_format_out](fd_out)
                sys.stdout.write('.')
                sys.stdout.flush()
#-------------------------------------------------------------------------------
    def diff(self):
        """
        Main-loop for difference calculations
        """
        out = self.filename.split('.')[0] + '_' + self.filename_diff.split('.')[0] + '_diff.yuv'
        fd_out = open(out, 'wb')
        with open(self.filename, 'rb') as fd_1, open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                self.reader[self.yuv_format_in](fd_1)
                data1 = self._g3(self.y)
                self.reader[self.yuv_format_in](fd_2)
                data2 = self._g3(self.y)

                D = []
                for x, y in zip(data1, data2):
                    D.append(self._clip(0x80-abs(x-y)))

                fd_out.write(self._g1(D))

                sys.stdout.write('.')
                sys.stdout.flush()
        fd_out.close()
#-------------------------------------------------------------------------------
    def _read_YV12(self, fd):
        """
        Read one frame
        """
        self.y = fd.read(self.width * self.height)
        self.cb = fd.read(self.width * self.height / 4)
        self.cr = fd.read(self.width * self.height / 4)
#-------------------------------------------------------------------------------
    def _read_UYVY(self, fd):
        """
        Read one frame
        """
        raw = fd.read(self.frame_size_in)

        self.y = raw[1::2]     # y
        self.cb = raw[0::4]    # u
        self.cr = raw[2::4]    # v
#-------------------------------------------------------------------------------
    def _read_YVYU(self, fd):
        """
        Read one frame
        """
        raw = fd.read(self.frame_size_in)

        # Y0|V0|Y1|U0
        self.y = raw[0::2]     # y
        self.cr = raw[1::4]    # v
        self.cb = raw[3::4]    # u
#-------------------------------------------------------------------------------
    def _read_IYUV(self, fd):
        """
        read one frame
        IYUV is YV12 where the croma-planes have switched places.
        """
        self.y = fd.read(self.width * self.height)
        self.cr = fd.read(self.width * self.height / 4)
        self.cb = fd.read(self.width * self.height / 4)
#-------------------------------------------------------------------------------
    def _write_YVYU(self, fd):
        """
        write one frame
        handle re-sampling of frame from 4:2:0 -> 4:2:2
        """

        if self.yuv_format_in in ['YV12', 'IYUV']:
            self.y = self._g3(self.y)
            self.cb = self._g3(self.cb)
            self.cr = self._g3(self.cr)
            cb = [0 for i in xrange(0, ((self.width >> 1) * self.height))]
            cr = [0 for i in xrange(0, ((self.width >> 1) * self.height))]

            self.cb = self._conv420to422(self.cb, cb)
            self.cr = self._conv420to422(self.cr, cr)

        yvyu = self.zero
        yvyu[0::2] = self.y
        yvyu[1::4] = self.cr
        yvyu[3::4] = self.cb

        if self.yuv_format_in in ['YV12', 'IYUV']:
            fd.write(self._g1(yvyu))
        else:
            fd.write("".join(yvyu))
#-------------------------------------------------------------------------------
    def _write_YV12(self, fd):
        """
        write one frame
        handle re-sampling of frame from 4:2:2 -> 4:2:0
        """
        if self.yuv_format_in in ['UYVY', 'YVYU']:
            self.y = self._g3(self.y)
            self.cb = self._g3(self.cb)
            self.cr = self._g3(self.cr)
            cb = [0 for i in xrange(0, ((self.width >> 2) * self.height))]
            cr = [0 for i in xrange(0, ((self.width >> 2) * self.height))]

            self.cb = self._conv422to420(self.cb, cb)
            self.cr = self._conv422to420(self.cr, cr)

            fd.write(self._g1(self.y) + self._g1(self.cb) + self._g1(self.cr))
        else:
            fd.write(self.y + self.cb + self.cr)

#-------------------------------------------------------------------------------
    def _write_UYVY(self, fd):
        """
        write one frame
        handle re-sampling of frame from 4:2:0 -> 4:2:2
        """
        if self.yuv_format_in in ['YV12', 'IYUV']:
            self.y = self._g3(self.y)
            self.cb = self._g3(self.cb)
            self.cr = self._g3(self.cr)
            cb = [0 for i in xrange(0, ((self.width >> 1) * self.height))]
            cr = [0 for i in xrange(0, ((self.width >> 1) * self.height))]

            self.cb = self._conv420to422(self.cb, cb)
            self.cr = self._conv420to422(self.cr, cr)

        uyvy = self.zero
        uyvy[1::2] = self.y
        uyvy[0::4] = self.cb
        uyvy[2::4] = self.cr

        if self.yuv_format_in in ['YV12', 'IYUV']:
            fd.write(self._g1(uyvy))
        else:
            fd.write("".join(uyvy))
#-------------------------------------------------------------------------------
    def _write_IYUV(self, fd):
        """
        write one frame
        handle re-sampling of frame from 4:2:2 -> 4:2:0
        """
        if self.yuv_format_in in ['UYVY', 'YVYU']:
            self.y = self._g3(self.y)
            self.cb = self._g3(self.cb)
            self.cr = self._g3(self.cr)
            cb = [0 for i in xrange(0, ((self.width >> 2) * self.height))]
            cr = [0 for i in xxrange(0, ((self.width >> 2) * self.height))]

            self.cb = self._conv422to420(self.cb, cb)
            self.cr = self._conv422to420(self.cr, cr)

            fd.write(self._g1(self.y) + self._g1(self.cr) + self._g1(self.cb))
        else:
            fd.write(self.y + self.cr + self.cb)
#-------------------------------------------------------------------------------
    def _conv420to422(self, src, dst):
        """
        420 to 422 - vertical 1:2 interpolation filter

        http://www.mpeg.org/MPEG/video/mssg-free-mpeg-software.html
        """
        w = self.width >> 1
        h = self.height >> 1

        n = 0
        k = 0
        for i in xrange(w):
            for j in xrange(h):
                j2 = j<<1
                jm3 = 0 if (j<3) else j-3
                jm2 = 0 if (j<2) else j-2
                jm1 = 0 if (j<1) else j-1
                jp1 = j+1 if (j<h-1) else h-1
                jp2 = j+2 if (j<h-2) else h-1
                jp3 = j+3 if (j<h-3) else h-1

                a = (3*src[n+w*jm3]
                   -16*src[n+w*jm2]
                   +67*src[n+w*jm1]
                  +227*src[n+w*j]
                   -32*src[n+w*jp1]
                    +7*src[n+w*jp2]+128)>>8
                dst[k+w*j2] = self._clip(a)

                b = (3*src[n+w*jp3]
                   -16*src[n+w*jp2]
                   +67*src[n+w*jp1]
                  +227*src[n+w*j]
                   -32*src[n+w*jm1]
                   +7*src[n+w*jm2]+128)>>8
                dst[k+w*(j2+1)] = self._clip(b)
            n += 1
            k += 1
        return dst
#-------------------------------------------------------------------------------
    def _conv422to420(self, src, dst):
        """
        422 -> 420

        http://www.mpeg.org/MPEG/video/mssg-free-mpeg-software.html
        """
        w = self.width >> 1
        h = self.height

        n=0
        k=0
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
                jp6 = j+6 if (j<h-6) else h-1

                # FIR filter with 0.5 sample interval phase shift
                a = ( 228*(src[n+w*j]  +src[n+w*jp1])
                      +70*(src[n+w*jm1]+src[n+w*jp2])
                      -37*(src[n+w*jm2]+src[n+w*jp3])
                      -21*(src[n+w*jm3]+src[n+w*jp4])
                      +11*(src[n+w*jm4]+src[n+w*jp5])
                       +5*(src[n+w*jm5]+src[n+w*jp6])+256)>>9
                dst[k+w*(j>>1)] = self._clip(a)
            n+=1
            k+=1
        return dst
#-------------------------------------------------------------------------------
    def _g3(self, string):
        """
        return a array of int from a string
        """
        #return [int(binascii.hexlify(i), 16) for i in string]
        return array.array('B', string).tolist()
        #return map(ord, string)
#-------------------------------------------------------------------------------
    def _g1(self, my_list):
        """create a string of integer ASCII values from a list of int"""
        return array.array('B', my_list).tostring()
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
        Split a file into separate frames
        """
        src_yuv = open(self.filename, 'rb')

        filecnt = 0
        while True:
            buf = src_yuv.read(self.frame_size_in)
            if buf:
                s = "frame" + "%s" % filecnt + ".yuv"
                dst_yuv = open(s, 'wb')
                dst_yuv.write(buf)           # write read data into new file
                print "writing frame", filecnt
                dst_yuv.close()
                filecnt += 1
            else:
                break
        src_yuv.close()
#-------------------------------------------------------------------------------
def my_test():
    """Add a test-suite here"""
#    x = YCbCr(width=352, height=288, filename='foreman_352x288.yuv', yuv_format_in='YV12')
#    x.show()
#    y = YCbCr(width=1600, height=1200, filename='image0.ycbcr', yuv_format_in='UYVY')
#    y.show()
#    a = YCbCr(width=352, height=288, filename='foreman_352x288.yuv', yuv_format_in='YV12',
#            yuv_format_out='UYVY', filename_out='out_UYVY.yuv')
#    a.show()
#    a.convert()
#
#    b = YCbCr(width=352, height=288, filename='foreman_352x288.yuv', yuv_format_in='YV12',
#            yuv_format_out='IYUV', filename_out='out_IYUV.yuv')
#    b.show()
#    b.convert()

    c = YCbCr(width=352, height=288, filename='foreman_352x288.yuv', yuv_format_in='YV12',
            yuv_format_out='YVYU', filename_out='out_YVYU.yuv')
    c.show()
    c.convert()
#    q = YCbCr(width=1600, height=1200, filename='image0.ycbcr', yuv_format_in='UYVY',
#            yuv_format_out='IYUV', filename_out='out.yuv')
#    q.show()
#    q.convert()
#    d = YCbCr(width=352, height=288, filename='foreman_352x288.yuv', yuv_format_in='YV12',
#            yuv_format_out='YV12', filename_out='out_YV12.yuv')
#    d.show()
#    d.convert()
#    e = YCbCr(width=352, height=288, filename='out_UYVY.yuv', yuv_format_in='UYVY',
#            yuv_format_out='UYVY', filename_out='slask.yuv')
#    e.show()
#    e.convert()
#-------------------------------------------------------------------------------

# helper functions
def cmd_info(args):
    YCbCr(**vars(args)).show()

def cmd_split(args):
    Y = YCbCr(**vars(args))
    Y.show()
    Y.split()

def cmd_convert(args):
    Y = YCbCr(**vars(args))
    Y.show()
    Y.convert()

def cmd_diff(args):
    Y = YCbCr(**vars(args))
    Y.show()
    Y.diff()

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
            choices=['IYUV', 'UYVY', 'YV12', 'YVYU'], help='valid input-formats')

    # create parser for the 'info' command
    parser_info = subparsers.add_parser('info', help='Basic info about YCbCr file',
            parents=[parent_parser])
    parser_info.set_defaults(func=cmd_info)

    # create parser for the "split" command
    parser_split = subparsers.add_parser('split',
            help='Split a YCbCr file into individual frames',
            parents=[parent_parser])
    parser_split.set_defaults(func=cmd_split)

    # create parser for the "convert" command
    parser_convert = subparsers.add_parser('convert',
            help='YCbCr format conversion',
            parents=[parent_parser])
    parser_convert.add_argument('yuv_format_out', type=str,
            choices=['IYUV', 'UYVY', 'YV12', 'YVYU'],
            help='valid output-formats')
    parser_convert.add_argument('filename_out', type=str, help='file to write to')
    parser_convert.set_defaults(func=cmd_convert)

    # create parser for the "diff" command
    parser_diff = subparsers.add_parser('diff',
            help='Create diff between two YCbCr files',
            parents=[parent_parser])
    parser_diff.add_argument('filename_diff', type=str, help='filename')
    parser_diff.set_defaults(func=cmd_diff)

    #let parse_args() do the job of calling the appropriate function
    # after argument parsing is complete
    args = parser.parse_args()
    args.func(args)
