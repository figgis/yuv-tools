YCbCr (YUV)
===========

Tools for various operations on raw YCbCr video files.
http://en.wikipedia.org/wiki/YCbCr

ycbcr.py - is the main class that supports the following formats:

* IYUV
* UYVY
* YV12
* YVYU
* YUY2
* 422

Supported operations:

* basic info about a file
* convert between any of the formats above (including correct sub- re-sampling of chroma-data)
* split a file into individual frames
* creates a diff between two files
* PSNR calculations, one value per color-plane
* get luma-data per frame
* SSIM calculation on luma


Usage
-----

	$ ./ycbcr.py info foreman_352x288.yuv 352 288 YV12
	$ ./ycbcr.py convert --help
	$ ./ycbcr.py diff --help
	$ ./ycbcr.py split --help
	$ ./ycbcr.py psnr --help
	$ ./ycbcr.py ssim --help
	$ ./plot_diff.py foreman_cif_frame_0.yuv foreman_cif_frame_1.yuv 352 288 YV12

Files
-----

* ycbcr.py - main class
* plot_diff.py - matplotlib wrapper around PSNR/SSIM-calculation. Generate nice plots using luma-data.
* verify.py - unittest

