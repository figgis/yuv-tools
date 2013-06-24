#!/usr/bin/env python

"""
Visually show various metrics between two YCbCr sequences
using matplotlib
"""

import argparse
import time
import os

import numpy as np
import matplotlib.pyplot as plt
from ycbcr import YCbCr

def create_title_string(arg):
    """
    Helper function to generate a nice looking
    title string
    """
    return "{} {}\n{}".format(
        os.path.basename(arg.filename),
        'VS.',
        os.path.basename(arg.filename_diff))

def plot_psnr(arg):
    """
    PSNR
    """
    yuv = YCbCr(**vars(arg))

    psnr = [p[0] for p in yuv.psnr()]

    N = len(psnr)
    ind = np.arange(N)  # the x locations for the groups

    plt.figure()
    plt.title(create_title_string(arg))
    plt.plot(ind, psnr, 'ro-')
    plt.ylabel('dB')
    plt.xlabel('frame')
    plt.grid(True)

    plt.show()

def plot_psnr_all(arg):
    """
    PSNR, all planes
    """
    yuv = YCbCr(**vars(arg))

    psnr = [p for p in yuv.psnr()]

    N = len(psnr)
    ind = np.arange(N)  # the x locations for the groups

    plt.figure()
    plt.title(create_title_string(arg))
    plt.plot(ind, [i[0] for i in psnr], 'ko-', label='Y')
    plt.plot(ind, [i[1] for i in psnr], 'bo-', label='Cb')
    plt.plot(ind, [i[2] for i in psnr], 'ro-', label='Cr')
    plt.plot(ind, [i[3] for i in psnr], 'mo-', label='Frame')
    plt.legend()
    plt.ylabel('dB')
    plt.xlabel('frame')
    plt.grid(True)

    plt.show()

def plot_ssim(arg):
    """
    SSIM
    """
    yuv = YCbCr(**vars(arg))

    ssim = [s for s in yuv.ssim()]

    N = len(ssim)
    ind = np.arange(N)  # the x locations for the groups

    plt.figure()
    plt.title(create_title_string(arg))
    plt.plot(ind, ssim, 'ro-')
    plt.ylabel('Index')
    plt.xlabel('frame')
    plt.grid(True)

    plt.show()

def main():
    """
    pass
    """

    # create the top-level parser
    parser = argparse.ArgumentParser(
        description='YCbCr visualization',
        epilog='Be careful with those bits')
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
    parent_parser.add_argument('filename_diff', type=str, help='filename')
    parent_parser.add_argument(
        '--num',
        type=int,
        default=None,
        help='number of frames to process [0..n-1]')

    # create parser for the 'psnr' command
    parser_psnr = subparsers.add_parser(
        'psnr',
        help='Luma PSNR plot',
        parents=[parent_parser])
    parser_psnr.set_defaults(func=plot_psnr)

    # create parser for the 'psnr_all' command
    parser_psnr_all = subparsers.add_parser(
        'psnr_all',
        help='PSNR plot, all planes',
        parents=[parent_parser])
    parser_psnr_all.set_defaults(func=plot_psnr_all)

    # create parser for the 'ssim' command
    parser_psnr = subparsers.add_parser(
        'ssim',
        help='Calculate ssim for each frame, luma data only',
        parents=[parent_parser])
    parser_psnr.set_defaults(func=plot_ssim)

    # let parse_args() do the job of calling the appropriate function
    # after argument parsing is complete
    args = parser.parse_args()
    t1 = time.clock()
    args.func(args)
    t2 = time.clock()
    print "\nTime: ", round(t2 - t1, 4)

if __name__ == '__main__':
    main()
