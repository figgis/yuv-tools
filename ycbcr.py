#!/usr/bin/env python

"""
Tools for working with YCbCr data.
"""

import argparse
import time
import sys
import os

from collections import namedtuple

import numpy as np


class Y:
    """
    BASE
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.wh = self.width * self.height
        self.div = namedtuple('chroma_div', 'width height')

    def get_420_partitioning(self, width=None, height=None):
        if not width:
            wh = self.wh
        else:
            wh = width * height
        # start-stop
        #       y  y   cb  cb      cr      cr
        return (0, wh, wh, wh/4*5, wh/4*5, wh/2*3)

    def get_422_partitioning(self, width=None, height=None):
        if not width:
            wh = self.wh
        else:
            wh = width * height
        # start-stop
        #       y  y   cb  cb      cr      cr
        return (0, wh, wh, wh/2*3, wh/2*3, wh*2)


class YV12(Y):
    """
    YV12
    """
    def __init__(self, width, height):
        Y.__init__(self, width, height)

        # width, height
        self.chroma_div = self.div(2, 2)  # Chroma divisor w.r.t luma-size

    def get_frame_size(self, width=None, height=None):
        if not width:
            width = self.width
            height = self.height
        return (width * height * 3 / 2)

    def get_layout(self, width=None, height=None):
        """
        return a tuple of slice-objects
        Y|U|V
        """
        p = self.get_420_partitioning(width, height)
        return (slice(p[0], p[1]),    # start-stop for luma
                slice(p[2], p[3]),    # start-stop for chroma
                slice(p[4], p[5]))    # start-stop for chroma


class IYUV(Y):
    """
    IYUV
    """
    def __init__(self, width, height):
        Y.__init__(self, width, height)
        self.chroma_div = self.div(2, 2)

    def get_frame_size(self, width=None, height=None):
        if not width:
            width = self.width
            height = self.height
        return (width * height * 3 / 2)

    def get_layout(self, width=None, height=None):
        """
        Y|V|U
        """
        p = self.get_420_partitioning(width, height)
        return (slice(p[0], p[1]),
                slice(p[4], p[5]),
                slice(p[2], p[3]))


class UYVY(Y):
    """
    UYVY
    """
    def __init__(self, width, height):
        Y.__init__(self, width, height)
        self.chroma_div = self.div(2, 1)

    def get_frame_size(self, width=None, height=None):
        if not width:
            width = self.width
            height = self.height
        return (width * height * 2)

    def get_layout(self, width=None, height=None):
        """
        U0|Y0|V0|Y1
        """
        fs = self.get_frame_size(width, height)
        return (slice(1, fs, 2),
                slice(0, fs, 4),
                slice(2, fs, 4))


class YVYU(Y):
    """
    YVYU
    """
    def __init__(self, width, height):
        Y.__init__(self, width, height)
        self.chroma_div = self.div(2, 1)

    def get_frame_size(self, width=None, height=None):
        if not width:
            width = self.width
            height = self.height
        return (width * height * 2)

    def get_layout(self, width=None, height=None):
        """
        Y0|V0|Y1|U0
        """
        fs = self.get_frame_size(width, height)
        return (slice(0, fs, 2),
                slice(3, fs, 4),
                slice(1, fs, 4))


class YUY2(Y):
    """
    YUY2
    """
    def __init__(self, width, height):
        Y.__init__(self, width, height)
        self.chroma_div = self.div(2, 1)

    def get_frame_size(self, width=None, height=None):
        if not width:
            width = self.width
            height = self.height
        return (width * height * 2)

    def get_layout(self, width=None, height=None):
        """
        Y0|U0|Y1|V0
        """
        fs = self.get_frame_size(width, height)
        return (slice(0, fs, 2),
                slice(1, fs, 4),
                slice(3, fs, 4))


class Y422(Y):
    """
    422
    """
    def __init__(self, width, height):
        Y.__init__(self, width, height)
        self.chroma_div = self.div(2, 1)

    def get_frame_size(self, width=None, height=None):
        if not width:
            width = self.width
            height = self.height
        return (width * height * 2)

    def get_layout(self, width=None, height=None):
        """
        Y|U|V
        """
        p = self.get_422_partitioning(width, height)
        return (slice(p[0], p[1]),
                slice(p[2], p[3]),
                slice(p[4], p[5]))


class Font:
    """
    pass
    """
    def __init__(self):
        self.char = (
            (0x7ffe, 0x7ffe, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x7ffe, 0x7ffe),  # 0
            (0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x0180, 0x1800, 0x0180),  # 1
            (0x7ffe, 0x7ffe, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x7ffe, 0x7ffe, 0x6000, 0x6000, 0x6000, 0x6000, 0x6000, 0x7ffe, 0x7ffe),  # 2
            (0x7ffe, 0x7ffe, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x7ffe, 0x7ffe, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x7ffe, 0x7ffe),  # 3
            (0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x7ffe, 0x7ffe, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006),  # 4
            (0x7ffe, 0x7ffe, 0x6000, 0x6000, 0x6000, 0x6000, 0x6000, 0x7ffe, 0x7ffe, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x7ffe, 0x7ffe),  # 5
            (0x6000, 0x6000, 0x6000, 0x6000, 0x6000, 0x6000, 0x6000, 0x7ffe, 0x7ffe, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x7ffe, 0x7ffe),  # 6
            (0x7ffe, 0x7ffe, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006),  # 7
            (0x7ffe, 0x7ffe, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x7ffe, 0x7ffe, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x7ffe, 0x7ffe),  # 8
            (0x7ffe, 0x7ffe, 0x6006, 0x6006, 0x6006, 0x6006, 0x6006, 0x7ffe, 0x7ffe, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006, 0x0006),  # 9
        )

    def show(self, num):
        if 0 < num > 9:
            return

        for c in self.char[num]:
            for i in range(15, -1, -1):
                if c & (1 << i):
                    sys.stdout.write("X")
                else:
                    sys.stdout.write(" ")
            print


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
    def __init__(
        self,
        width=0,
        height=0,
        filename=None,
        yuv_format_in=None,
        yuv_format_out=None,
        filename_out=None,
        filename_diff=None,
        crop_rect=None,
        num=None,
        func=None):

        self.supported_420 = [
            'YV12',
            'IYUV',
        ]

        self.supported_422 = [
            'UYVY',
            'YVYU',
            'YUY2',
            '422',
        ]

        self.supported_extra = [
            None,
        ]

        if yuv_format_in not in self.supported_420 + self.supported_422 + \
           self.supported_extra:
            raise NameError('Format not supported! "%s"' % yuv_format_in)

        if yuv_format_out not in self.supported_420 + self.supported_422 + \
           self.supported_extra:
            raise NameError('Format not supported! "%s"' % yuv_format_out)

        self.filename = filename
        self.filename_out = filename_out
        self.filename_diff = filename_diff
        self.width = width
        self.height = height
        self.yuv_format_in = yuv_format_in
        self.yuv_format_out = yuv_format_out

        if crop_rect:
            rect = namedtuple('rect', 'xs ys xe ye')
            self.crop_rect = rect(*crop_rect)

        self.yy = None
        self.cb = None
        self.cr = None

        # Reader/Writer
        RW = {
            'YV12': YV12,
            'IYUV': IYUV,
            'UYVY': UYVY,
            'YVYU': YVYU,
            'YUY2': YUY2,
            '422': Y422,
        }

        # Setup
        if self.yuv_format_in:  # we need a reader and and a writer just
                                # to make sure
            self.reader = RW[self.yuv_format_in](self.width, self.height)
            self.writer = RW[self.yuv_format_in](self.width, self.height)
            self.frame_size_in = self.reader.get_frame_size()
            self.frame_size_out = self.reader.get_frame_size()
            self.num_frames = os.path.getsize(self.filename) / self.frame_size_in
            self.layout_in = self.reader.get_layout()
            self.layout_out = self.reader.get_layout()
            self.frame_size_out = self.frame_size_in
            self.chroma_div = self.reader.chroma_div

        if self.yuv_format_out:
            self.writer = RW[self.yuv_format_out](self.width, self.height)
            self.frame_size_out = self.writer.get_frame_size()
            self.layout_out = self.writer.get_layout()

        # 8bpp -> 10bpp, 10->8 dito; special handling
        if yuv_format_in is not None:
            self.__check()

        # How many frames to process
        if num:
            if num <= self.num_frames:
                self.num_frames = num

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
        print "Num frames:", os.path.getsize(self.filename) / self.frame_size_in
        print "Size of 1 frame (in) (bytes):", self.frame_size_in
        print "Size of 1 frame (out) (bytes):", self.frame_size_out
        print

    def convert(self):
        """
        Format-conversion between the supported formats.
        4:2:0 to 4:2:2 interpolation and 4:2:2 to 4:2:0
        subsampling when necessary.
        """
        self.__execute()

    def diff(self):
        """
        Produces a YV12 file containing the luma-difference between
        two files.
        """
        base1 = os.path.basename(self.filename)
        base2 = os.path.basename(self.filename_diff)
        out = os.path.splitext(base1)[0] + '_' + \
            os.path.splitext(base2)[0] + '_diff.yuv'

        chroma = np.empty(self.width * self.height / 2, dtype=np.uint8)
        chroma.fill(0x80)
        fd_out = open(out, 'wb')
        with open(self.filename, 'rb') as fd_1, \
                open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_1)
                data1 = self.yy.copy()
                self.__read_frame(fd_2)
                data2 = self.yy.copy()

                data = 0x80 - np.abs(data1 - data2)
                data = data.astype(np.uint8, copy=False)
                data.tofile(fd_out)
                chroma.tofile(fd_out)
                sys.stdout.write('.')
                sys.stdout.flush()
        fd_out.close()

    def psnr(self):
        """
        PSNR calculations.
        Generator gives PSNR for
        [Y, Cb, Cr, whole frame]

        http://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        """
        def psnr(a, b):
            m = ((a - b) ** 2).mean()
            if m == 0:
                return float("nan")

            return 10 * np.log10(255 ** 2 / m)

        with open(self.filename, 'rb') as fd_1, \
                open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_1)
                frame1 = self.__copy_planes()
                self.__read_frame(fd_2)
                frame2 = self.__copy_planes()

                yield [psnr(x, y) for x, y in zip(frame1, frame2)]

    def wpsnr(self):
        """
        Calculate the Weighted sum of the PSNR
        per picture of the individual components
        http://iphome.hhi.de/wiegand/assets/pdfs/2012_12_IEEE-HEVC-Performance.pdf
        p.1676
        """
        for i in self.psnr():
            yield (6 * i[0] + i[1] + i[2]) / 8.0

    def wpsnravg(self):
        """
        Return the average of the Weighted sum of the PSNR
        """
        return np.average(np.fromiter((i for i in self.wpsnr()), np.float))

    def ssim(self):
        """
        http://en.wikipedia.org/wiki/Structural_similarity

        implementation using scipy and numpy from
        http://isit.u-clermont1.fr/~anvacava/code.html
        by antoine.vacavant@udamail.fr
        Usage by kind permission from author.
        """
        import scipy.ndimage
        from numpy.ma.core import exp
        from scipy.constants.constants import pi

        def compute_ssim(img_mat_1, img_mat_2):
            #Variables for Gaussian kernel definition
            gaussian_kernel_sigma = 1.5
            gaussian_kernel_width = 11
            gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

            #Fill Gaussian kernel
            for i in range(gaussian_kernel_width):
                for j in range(gaussian_kernel_width):
                    gaussian_kernel[i, j] = \
                        (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) *\
                        exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

            #Convert image matrices to double precision (like in the Matlab version)
            img_mat_1 = img_mat_1.astype(np.float)
            img_mat_2 = img_mat_2.astype(np.float)

            #Squares of input matrices
            img_mat_1_sq = img_mat_1 ** 2
            img_mat_2_sq = img_mat_2 ** 2
            img_mat_12 = img_mat_1 * img_mat_2

            #Means obtained by Gaussian filtering of inputs
            img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
            img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

            #Squares of means
            img_mat_mu_1_sq = img_mat_mu_1 ** 2
            img_mat_mu_2_sq = img_mat_mu_2 ** 2
            img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2

            #Variances obtained by Gaussian filtering of inputs' squares
            img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
            img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)

            #Covariance
            img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)

            #Centered squares of variances
            img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
            img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
            img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

            #c1/c2 constants
            #First use: manual fitting
            c_1 = 6.5025
            c_2 = 58.5225

            #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
            l = 255
            k_1 = 0.01
            c_1 = (k_1 * l) ** 2
            k_2 = 0.03
            c_2 = (k_2 * l) ** 2

            #Numerator of SSIM
            num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
            #Denominator of SSIM
            den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) *\
                (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
            #SSIM
            ssim_map = num_ssim / den_ssim
            index = np.average(ssim_map)

            return index

        with open(self.filename, 'rb') as fd_1, \
                open(self.filename_diff, 'rb') as fd_2:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_1)
                data1 = self.yy.copy()
                self.__read_frame(fd_2)
                data2 = self.yy.copy()

                yield compute_ssim(np.reshape(data1, (self.height, self.width)),
                                   np.reshape(data2, (self.height, self.width)))

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
                yield self.yy

    def split(self):
        """
        Split a file into separate frames.
        """
        src_yuv = open(self.filename, 'rb')

        for i in xrange(self.num_frames):
            data = src_yuv.read(self.frame_size_in)
            fname = "frame" + "%d" % i + ".yuv"
            dst_yuv = open(fname, 'wb')
            dst_yuv.write(data)
            sys.stdout.write('.')
            sys.stdout.flush()
            dst_yuv.close()
        src_yuv.close()

    def eight2ten(self):
        """
        8 bpp -> 10 bpp
        """
        a_in = np.memmap(self.filename, mode='readonly')
        a_out = np.memmap(self.filename_out, mode='write', shape=2 * len(a_in))
        a_out[::2] = a_in << 2
        a_out[1::2] = a_in >> 6

    def ten2eight(self):
        """
        10 bpp -> 8 bpp
        """
        fd_i = open(self.filename, 'rb')
        fd_o = open(self.filename_out, 'wb')

        while True:
            chunk = np.fromfile(fd_i, dtype=np.uint8, count=8192)
            chunk = chunk.astype(np.uint)
            if not chunk.any():
                break
            data = (2 + (chunk[1::2] << 8 | chunk[0::2])) >> 2

            data = data.astype(np.uint8, casting='same_kind')
            data.tofile(fd_o)

        fd_i.close()
        fd_o.close()

    def fliplr(self):
        """
        Flip left-right
        """
        d = self.chroma_div
        with open(self.filename, 'rb') as fd_in, \
                open(self.filename_out, 'wb') as fd_out:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_in)
                x = self.yy.reshape([self.height, self.width])
                self.yy = np.fliplr(x).reshape(-1)

                x = self.cb.reshape([self.height / d.height, self.width / d.width])
                self.cb = np.fliplr(x).reshape(-1)

                x = self.cr.reshape([self.height / d.height, self.width / d.width])
                self.cr = np.fliplr(x).reshape(-1)

                self.__write_frame(fd_out)
                sys.stdout.write('.')
                sys.stdout.flush()

    def flipud(self):
        """
        Flip upside-down
        """
        with open(self.filename, 'rb') as fd_in, \
                open(self.filename_out, 'wb') as fd_out:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_in)
                self.yy = np.flipud(self.yy)
                self.cb = np.flipud(self.cb)
                self.cr = np.flipud(self.cr)
                self.__write_frame(fd_out)
                sys.stdout.write('.')
                sys.stdout.flush()

    def draw_frame_number(self):
        """
        Draw frame-number in Luma-data
        """
        drawer = Font()
        self.__execute(self.__add_frame_number, drawer)

    def crop(self):
        """
        Crop
        """
        c = self.crop_rect
        w = c[2] - c[0] + 1
        h = c[3] - c[1] + 1
        # re-calculate output-layout
        self.frame_size_out = self.reader.get_frame_size(w, h)
        self.layout_out = self.reader.get_layout(w, h)

        self.__execute(self.__crop)


    def reduce_framerate(self, fin, fout):
        """
        Reduce framerate by throwing frames away
        input:  filename.yuv
        output: filename_xx_fps.yuv
        """
        fname, fext = os.path.splitext(os.path.basename(self.filename))
        fname_out = "%s_%d_fps%s" % (fname, fout, fext)

        print "Writing result to", os.getcwd(), fname_out

        with open(self.filename, 'rb') as fd_1, \
                open(fname_out, 'wb') as fd_2:
            for i in xrange(self.num_frames):
                data = fd_1.read(self.frame_size_in)
                if i%(fin/fout) == 0:
                    fd_2.write(data)
                sys.stdout.write('.')
                sys.stdout.flush()

    def __execute(self, func=lambda *a, **k: None, *args, **kwargs):
        """
        Wrapper around read/write frame
        """
        with open(self.filename, 'rb') as fd_in, \
                open(self.filename_out, 'wb') as fd_out:
            for i in xrange(self.num_frames):
                self.__read_frame(fd_in)
                func(i, *args, **kwargs)
                self.__write_frame(fd_out)
                sys.stdout.write('.')
                sys.stdout.flush()

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
            if not os.path.getsize(self.filename) == \
               os.path.getsize(self.filename_diff):
                print >> sys.stderr, "[WARNING] - file-sizes are not equal"

    def __read_frame(self, fd):
        """
        Use extended indexing to read 1 frame into self.{y, cb, cr}
        """
        self.raw = np.fromfile(fd, dtype=np.uint8, count=self.frame_size_in)
        self.raw = self.raw.astype(np.int, copy=False)

        self.yy = self.raw[self.layout_in[0]]
        self.cb = self.raw[self.layout_in[1]]
        self.cr = self.raw[self.layout_in[2]]

    def __write_frame(self, fd):
        """
        Use extended indexing to write 1 frame, including re-sampling and
        format conversion
        """
        self.__resample()
        data = np.empty(self.frame_size_out, dtype=np.uint8)

        data[self.layout_out[0]] = self.yy
        data[self.layout_out[1]] = self.cb
        data[self.layout_out[2]] = self.cr

        data.tofile(fd)

    def __resample(self):
        """
        Handle 420 -> 422 and 422 -> 420
        """

        if self.yuv_format_in in self.supported_420 and \
           self.yuv_format_out in self.supported_422:
            cb = np.zeros(self.width * self.height / 2, dtype=np.int)
            cr = np.zeros(self.width * self.height / 2, dtype=np.int)

            self.cb = self.__conv420to422(self.cb, cb)
            self.cr = self.__conv420to422(self.cr, cr)

        if self.yuv_format_in in self.supported_422 and \
           self.yuv_format_out in self.supported_420:
            cb = np.zeros(self.width * self.height / 4, dtype=np.int)
            cr = np.zeros(self.width * self.height / 4, dtype=np.int)

            self.cb = self.__conv422to420(self.cb, cb)
            self.cr = self.__conv422to420(self.cr, cr)

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

    def __conv422to420(self, src, dst):
        """
        422 -> 420

        http://www.mpeg.org/MPEG/video/mssg-free-mpeg-software.html
        although reference implementation reads data out-of-bounds,
        jp6 is the offending parameter. linking with electric-fence
        core-dumps. Bit-exact after change.
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

    def __rgb2ycbcr(self, r, g, b):
        """
        (r,g,b) -> (y, cb, cr)

        Conversion to YCbCr color space.
        CCIR 601 formulas from "Digital Pictures by Natravali and Haskell, page 120.
        """
        y = self.__clip2UInt8(0.257 * r + 0.504 * g + 0.098 * b + 16)
        cb = self.__clip2UInt8(-0.148 * r - 0.291 * g + 0.439 * b + 128)
        cr = self.__clip2UInt8(0.439 * r - 0.368 * g - 0.071 * b + 128)

        return (y, cb, cr)

    def __ycbcr2rgb(self, y, cb, cr):
        """
        (y,cb,cr) -> (r, g, b)

        Conversion to RGB color space.
        CCIR 601 formulas from "Digital Pictures by Natravali and Haskell, page 120.
        """
        y = y - 16
        cb = cb - 128
        cr = cr - 128

        r = self.__clip2UInt8(1.164 * y + 1.596 * cr)
        g = self.__clip2UInt8(1.164 * y - 0.392 * cb - 0.813 * cr)
        b = self.__clip2UInt8(1.164 * y + 2.017 * cb)

        return (r, g, b)

    def __clip2UInt8(self, d):
        "Clip d to interval 0-255"

        if (d < 0):
            return 0

        if (d > 255):
            return 255

        return int(round(d))

    def __copy_planes(self):
        """
        Return a copy of the different color planes,
        including whole frame
        """
        return self.yy.copy(), self.cb.copy(), self.cr.copy(), self.raw.copy()

    def __add_frame_number(self, frame, D):
        """
        Draw frame-number in Luma-data
        """
        self.yy = np.reshape(self.yy, (self.height, self.width))
        num_digits = map(int, str(frame))

        for pos, nd in enumerate(num_digits):

            digit = D.char[nd]

            for row, d in enumerate(digit):
                for i in range(15, -1, -1):
                    if d & (1 << i):
                        self.yy[row][pos*16:pos*16+16][15-i] = 16

        self.yy = self.yy.reshape(-1)

    def __crop(self, arg):
        """
        Crop color-planes
        Layout for 4:2:0
        X - Luma sample
        O - Co-located chroma sample
            i.e. 1 chroma for 4 Luma

        X     X     X     X
           O           O
        X     X     X     X
        X     X     X     X
           O           O
        X     X     X     X

        Layout for 4:2:2

        X     X     X     X
        O           O
        X     X     X     X
        O           O
        X     X     X     X
        O           O
        X     X     X     X
        O           O
        """
        d = self.chroma_div    # divisor
        r = self.crop_rect

        self.yy = np.reshape(self.yy, (self.height, self.width))
        self.yy = self.yy[r.ys:r.ye + 1, r.xs:r.xe + 1]

        self.yy = self.yy.reshape(-1)

        self.cb = self.cb.reshape([self.height / d.height,
                                   self.width / d.width])
        self.cb = self.cb[r.ys / d.height:
                          r.ye / d.height + 1,
                          r.xs / d.width:
                          r.xe / d.width + 1]
        self.cb = self.cb.reshape(-1)

        self.cr = self.cr.reshape([self.height / d.height,
                                   self.width / d.width])
        self.cr = self.cr[r.ys / d.height:
                          r.ye / d.height + 1,
                          r.xs / d.width:
                          r.xe / d.width + 1]
        self.cr = self.cr.reshape(-1)

def main():
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
        print "{:<5} {:<10} {:<10} {:<10} {:<10}".format('#', 'Y', 'Cb', 'Cr', 'Frame')
        for i, n in enumerate(yuv.psnr()):
            print "{:<5} {:<10f} {:<10f} {:<10f} {:<10f}".format(i, *n)

    def __cmd_wpsnr(arg):
        yuv = YCbCr(**vars(arg))
        for i, n in enumerate(yuv.wpsnr()):
            print "{:<5} {:<10f}".format(i, n)

    def __cmd_wpsnr_avg(arg):
        yuv = YCbCr(**vars(arg))
        print yuv.wpsnravg()

    def __cmd_ssim(arg):
        yuv = YCbCr(**vars(arg))
        for i, n in enumerate(yuv.ssim()):
            print i, n

    def __cmd_get_luma(arg):
        yuv = YCbCr(**vars(arg))
        return yuv.get_luma()

    def __cmd_8to10(arg):
        yuv = YCbCr(**vars(arg))
        yuv.eight2ten()

    def __cmd_10to8(arg):
        yuv = YCbCr(**vars(arg))
        yuv.ten2eight()

    def __cmd_fliplr(arg):
        yuv = YCbCr(**vars(arg))
        yuv.fliplr()

    def __cmd_flipud(arg):
        yuv = YCbCr(**vars(arg))
        yuv.flipud()

    def __cmd_fnum(arg):
        yuv = YCbCr(**vars(arg))
        yuv.draw_frame_number()

    def __cmd_crop(arg):
        yuv = YCbCr(**vars(arg))
        yuv.crop()

    def __cmd_fr(arg):
        fin = arg.fr_in
        fout = arg.fr_out
        del vars(arg)['fr_in']
        del vars(arg)['fr_out']

        if int(fout) >= int(fin):
            print >> sys.stderr, "[ERROR] - New framerate must be less than original"
            return

        yuv = YCbCr(**vars(arg))
        yuv.show()
        yuv.reduce_framerate(int(fin), int(fout))

    def coords(s):
        """
        rect must be positive
        TODO: add arg-check
        """
        try:
            x, y, z, q = map(int, s.split(','))
            return x, y, z, q
        except:
            raise argparse.ArgumentTypeError("Coordinates must be x,y,z,q")

    # create the top-level parser
    parser = argparse.ArgumentParser(
        description='YCbCr tools',
        epilog=' Be careful with those bits')
    subparsers = parser.add_subparsers(
        title='subcommands',
        help='additional help')

    # parent, common arguments for functions
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('filename', type=str, help='filename')
    parent_parser.add_argument('width', type=int)
    parent_parser.add_argument('height', type=int)
    parent_parser.add_argument(
        'yuv_format_in', type=str,
        choices=['IYUV', 'UYVY', 'YV12', 'YVYU', 'YUY2', '422'],
        help='valid input-formats')
    parent_parser.add_argument(
        '--num',
        type=int,
        default=None,
        help='number of frames to process [0..n-1]')

    # create parser for the 'info' command
    parser_info = subparsers.add_parser(
        'info',
        help='Basic info about YCbCr file',
        parents=[parent_parser])
    parser_info.set_defaults(func=__cmd_info)

    # create parser for the 'split' command
    parser_split = subparsers.add_parser(
        'split',
        help='Split a YCbCr file into individual frames',
        parents=[parent_parser])
    parser_split.set_defaults(func=__cmd_split)

    # create parser for the 'convert' command
    parser_convert = subparsers.add_parser(
        'convert',
        help='YCbCr format conversion',
        parents=[parent_parser])
    parser_convert.add_argument(
        'yuv_format_out', type=str,
        choices=['IYUV', 'UYVY', 'YV12', 'YVYU', '422', 'YUY2'],
        help='valid output-formats')
    parser_convert.add_argument('filename_out', type=str,
                                help='file to write to')
    parser_convert.set_defaults(func=__cmd_convert)

    # create parser for the 'diff' command
    parser_diff = subparsers.add_parser(
        'diff',
        help='Create diff between two YCbCr files',
        parents=[parent_parser])
    parser_diff.add_argument('filename_diff', type=str, help='filename')
    parser_diff.set_defaults(func=__cmd_diff)

    # create parser for the 'psnr' command
    parser_psnr = subparsers.add_parser(
        'psnr',
        help='Calculate PSNR for each frame and each plane',
        parents=[parent_parser])
    parser_psnr.add_argument('filename_diff', type=str, help='filename')
    parser_psnr.set_defaults(func=__cmd_psnr)

    # create parser for the 'weighted psnr' command
    parser_wpsnr = subparsers.add_parser(
        'wpsnr',
        help='Calculate the weighted sum of the PSNR',
        parents=[parent_parser])
    parser_wpsnr.add_argument('filename_diff', type=str, help='filename')
    parser_wpsnr.set_defaults(func=__cmd_wpsnr)

    # create parser for the 'weighted psnr average' command
    parser_wpsnr_avg = subparsers.add_parser(
        'wpsnravg',
        help='Calculate the average of the weighted sum of the PSNR',
        parents=[parent_parser])
    parser_wpsnr_avg.add_argument('filename_diff', type=str, help='filename')
    parser_wpsnr_avg.set_defaults(func=__cmd_wpsnr_avg)

    # create parser for the 'ssim' command
    parser_psnr = subparsers.add_parser(
        'ssim',
        help='Calculate ssim for each frame, luma data only',
        parents=[parent_parser])
    parser_psnr.add_argument('filename_diff', type=str, help='filename')
    parser_psnr.set_defaults(func=__cmd_ssim)

    # create parser for the 'get_luma' command
    parser_info = subparsers.add_parser(
        'get_luma',
        help='Return luminance-data for each frame. Generator',
        parents=[parent_parser])
    parser_info.set_defaults(func=__cmd_get_luma)

    # create parser for the '8to10' command
    parser_8to10 = subparsers.add_parser('8to10',
                                         help='YCbCr 8bpp -> 10bpp')
    parser_8to10.add_argument('filename', type=str, help='filename')
    parser_8to10.add_argument('filename_out', type=str,
                              help='file to write to')
    parser_8to10.set_defaults(func=__cmd_8to10)

    # create parser for the '10to8' command
    parser_10to8 = subparsers.add_parser('10to8',
                                         help='YCbCr 10bpp -> 8bpp')
    parser_10to8.add_argument('filename', type=str, help='filename')
    parser_10to8.add_argument('filename_out', type=str,
                              help='file to write to')
    parser_10to8.set_defaults(func=__cmd_10to8)

    # create parser for the 'fliplr' command
    parser_fliplr = subparsers.add_parser(
        'fliplr',
        help='Flip left-right',
        parents=[parent_parser])
    parser_fliplr.add_argument('filename_out', type=str,
                               help='file to write to')
    parser_fliplr.set_defaults(func=__cmd_fliplr)

    # create parser for the 'flipud' command
    parser_flipud = subparsers.add_parser(
        'flipud',
        help='Flip upside-down',
        parents=[parent_parser])
    parser_flipud.add_argument('filename_out', type=str,
                               help='file to write to')
    parser_flipud.set_defaults(func=__cmd_flipud)

    # create parser for the 'fnum' command
    parser_fnum = subparsers.add_parser(
        'fnum',
        help='Add Frame number',
        parents=[parent_parser])
    parser_fnum.add_argument('filename_out', type=str,
                             help='file to write to')
    parser_fnum.set_defaults(func=__cmd_fnum)

    # create parser for the 'crop' command
    parser_crop = subparsers.add_parser(
        'crop',
        help='Crop',
        parents=[parent_parser])
    parser_crop.add_argument('filename_out', type=str,
                             help='file to write to')
    parser_crop.add_argument('crop_rect', type=coords,
                             help='crop vector: \
                             x_start, y_start, x_end, y_end. \
                             Top-left corner has coordinates (0,0). \
                             1st MB: 0,0,15,15 \
                             2nd MB: 16,0,31,15')
    parser_crop.set_defaults(func=__cmd_crop)

    # create parser for the 'framerate' command
    parser_fr = subparsers.add_parser(
        'fr',
        help='Descrease framerate by throwing away frames',
        parents=[parent_parser])
    parser_fr.add_argument('fr_in', type=int, help='Current framerate')
    parser_fr.add_argument('fr_out', type=int, help='Target framerate')
    parser_fr.set_defaults(func=__cmd_fr)

    # let parse_args() do the job of calling the appropriate function
    # after argument parsing is complete
    args = parser.parse_args()
    t1 = time.clock()
    args.func(args)
    t2 = time.clock()
    print "\nTime: ", round(t2 - t1, 4)

if __name__ == '__main__':
    main()
