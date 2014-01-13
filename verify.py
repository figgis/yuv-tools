#!/usr/bin/env python

import unittest
import hashlib
import math

from ycbcr import YCbCr


SIZE_420 = 152064    # CIF w*h*3/2
SIZE_422 = 202752    # CIF w*h*2
OUT = 'slask.yuv'
OUT1 = 'slask1.yuv'


def get_sha1(f, size):
    """
    return sha1sum
    """
    with open(f, 'rb') as fd:
        buf = fd.read(size)
        return hashlib.sha1(buf).hexdigest()


class TestYCbCrFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_1(self):
        """
        convert YV12 into itself
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='YV12', filename_out=OUT)
        a.convert()

        ret = get_sha1(OUT, SIZE_420)

        self.assertEqual(ret, 'c9120ccf583b410b75379c48325dd50ec8d16ce8')

    def test_2(self):
        """
        YV12 -> UYVY
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='UYVY', filename_out=OUT)
        a.convert()

        ret = get_sha1(OUT, SIZE_422)

        self.assertEqual(ret, 'f50fc0500b217256a87c7cd1e867da0c49c51ace')

    def test_3(self):
        """
        YV12 -> YVYU
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='YVYU', filename_out=OUT)
        a.convert()

        ret = get_sha1(OUT, SIZE_422)

        self.assertEqual(ret, '68ac533290a89625b910731c93fbecba89b61870')

    def test_4(self):
        """
        YV12 -> UVYU -> YV12
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='UYVY', filename_out=OUT)
        a.convert()

        b = YCbCr(width=352, height=288, filename=OUT,
                  yuv_format_in='UYVY',
                  yuv_format_out='YV12', filename_out=OUT1)
        b.convert()

        ret = get_sha1(OUT1, SIZE_422)

        self.assertEqual(ret, 'b8934d77e0d71e77e90b4ba777a0cb978679d8ec')

    def test_5(self):
        """
        YV12 -> 422
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='422', filename_out=OUT)
        a.convert()

        ret = get_sha1(OUT, SIZE_422)

        self.assertEqual(ret, 'c700a31a209df30b72c1097898740d4c42d63a42')

    def test_6(self):
        """
        diff
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  filename_diff='foreman_cif_frame_1.yuv')
        a.diff()

        ret = get_sha1('foreman_cif_frame_0_foreman_cif_frame_1_diff.yuv', SIZE_420)

        self.assertEqual(ret, '6b508de1971eaae965d3a3cf0c8715c6fe907aff')

    def test_7(self):
        """
        split
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12')
        a.split()

        ret = get_sha1('frame0.yuv', SIZE_420)

        self.assertEqual(ret, 'c9120ccf583b410b75379c48325dd50ec8d16ce8')

    def test_8(self):
        """
        psnr
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12', filename_diff='foreman_cif_frame_1.yuv')

        ret = a.psnr().next()

        self.assertEqual(ret, [27.68336995961328, 43.025594686475237, 43.343456122199385, 31.558658820794285])

    def test_9(self):
        """
        ssim
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12', filename_diff='foreman_cif_frame_1.yuv')

        ret = a.ssim().next()

        self.assertEqual(ret, 0.8714863949031405)

    def test_10(self):
        """
        8bpp -> 10bpp
        """
        a = YCbCr(filename='foreman_cif_frame_0.yuv', filename_out='test_10.yuv')
        a.eight2ten()

        ret = get_sha1('test_10.yuv', SIZE_420 * 2)
        self.assertEqual(ret, '9cbade807771aa135f7f90b07e4bb510273b4e4f')

    def test_11(self):
        """
        10bpp -> 8bpp
        """
        a = YCbCr(filename='test_10.yuv', filename_out=OUT)
        a.ten2eight()

        ret = get_sha1(OUT, SIZE_420)
        self.assertEqual(ret, 'c9120ccf583b410b75379c48325dd50ec8d16ce8')

    def test_12(self):
        """
        YV12 -> IYUV
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='IYUV', filename_out=OUT)
        a.convert()

        ret = get_sha1(OUT, SIZE_420)

        self.assertEqual(ret, '385c87ed96b9298be9a410ff041fe4232b27a9aa')

    def test_13(self):
        """
        YV12 -> YUY2
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='YUY2', filename_out=OUT)
        a.convert()

        ret = get_sha1(OUT, SIZE_420)

        self.assertEqual(ret, '410b438c1aedc7e2ee6d68405f411bbfa8131b7a')

    def test_14(self):
        """
        psnr - nan
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12', filename_diff='foreman_cif_frame_0.yuv')

        ret = a.psnr().next()

        self.assertTrue(math.isnan(ret[0]))
        self.assertTrue(math.isnan(ret[1]))
        self.assertTrue(math.isnan(ret[2]))
        self.assertTrue(math.isnan(ret[3]))

    def test_15(self):
        """
        YV12 -> flip upside-down
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  filename_out=OUT)
        a.flipud()

        ret = get_sha1(OUT, SIZE_420)

        self.assertEqual(ret, '9052c6e03d7e4b8b2ec5d80aa17e9585b9b2a672')

    def test_16(self):
        """
        YV12 -> flip left-right
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  filename_out=OUT)
        a.fliplr()

        ret = get_sha1(OUT, SIZE_420)

        self.assertEqual(ret, 'f93ef579e5f672fd2a5962072509d98382a7d1d3')

    def test_17(self):
        """
        YV12 -> YVYU -> flip left-right
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='YVYU', filename_out=OUT)
        a.convert()

        a = YCbCr(width=352, height=288, filename=OUT,
                  yuv_format_in='YVYU',
                  filename_out=OUT1)
        a.fliplr()

        ret = get_sha1(OUT1, SIZE_422)

        self.assertEqual(ret, 'a5bdbfdeb0259f6d0bd85ddd97d58f6fc7965ca9')

    def test_18(self):
        """
        Drawing primitive
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  filename_out=OUT)
        a.draw_frame_number()

        ret = get_sha1(OUT, SIZE_420)

        self.assertEqual(ret, '71490f0356d3ba8d2c3d6110b77a9f665b561324')

    def test_19(self):
        """
        YV12 -> crop
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  crop_rect=(0,0,15,15),
                  filename_out=OUT)
        a.crop()

        ret = get_sha1(OUT, 384)

        self.assertEqual(ret, 'e74e61a5c6ade64c9f6371920512b5010d23cad4')

    def test_20(self):
        """
        YV12 -> YVYU -> crop
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='YVYU', filename_out=OUT)
        a.convert()

        a = YCbCr(width=352, height=288, filename=OUT,
                  yuv_format_in='YVYU',
                  crop_rect=(0, 0, 15, 15),
                  filename_out=OUT1)
        a.crop()

        ret = get_sha1(OUT1, 512)

        self.assertEqual(ret, '4bf3179e43e7ca6d490bf1d996dff84ef64be664')

    def test_21(self):
        """
        convert YV12 into NV12
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='NV12', filename_out=OUT)
        a.convert()

        ret = get_sha1(OUT, SIZE_420)

        self.assertEqual(ret, 'e24ac66dc32cff5dff16297dfaab761ab962143c')

if __name__ == '__main__':
    unittest.main()
