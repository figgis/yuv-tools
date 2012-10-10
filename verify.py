#!/usr/bin/env python

import unittest
import hashlib

from ycbcr import YCbCr

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

    def test_a(self):
        """
        convert YV12 into itself
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='YV12', filename_out='test_a.yuv')
        a.convert()

        ret = get_sha1('test_a.yuv', 152064)

        self.assertEqual(ret, 'c9120ccf583b410b75379c48325dd50ec8d16ce8')

    def test_b(self):
        """
        YV12 -> UYVY
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='UYVY', filename_out='test_b.yuv')
        a.convert()

        ret = get_sha1('test_b.yuv', 202752)

        self.assertEqual(ret, 'f50fc0500b217256a87c7cd1e867da0c49c51ace')

    def test_c(self):
        """
        YV12 -> YVYU
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='YVYU', filename_out='test_c.yuv')
        a.convert()

        ret = get_sha1('test_c.yuv', 202752)

        self.assertEqual(ret, '68ac533290a89625b910731c93fbecba89b61870')

    def test_d(self):
        """
        YV12 -> UVYU -> YV12
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                  yuv_format_in='YV12',
                  yuv_format_out='UYVY', filename_out='test_d.yuv')
        a.convert()

        b = YCbCr(width=352, height=288, filename='test_d.yuv', yuv_format_in='UYVY',
                  yuv_format_out='YV12', filename_out='test_d_2.yuv')
        b.convert()

        ret = get_sha1('test_d_2.yuv', 202752)

        self.assertEqual(ret, 'b8934d77e0d71e77e90b4ba777a0cb978679d8ec')

    def test_e(self):
        """
        YV12 -> 422
        """
        a = YCbCr(width=352, height=288, filename='foreman_cif_frame_0.yuv',
                yuv_format_in='YV12',
                yuv_format_out='422', filename_out='test_e.yuv')
        a.convert()

        ret = get_sha1('test_e.yuv', 202752)

        self.assertEqual(ret, 'c700a31a209df30b72c1097898740d4c42d63a42')
if __name__ == '__main__':
    unittest.main()
