YCbCr (YUV)
===========

Tools for various operations on raw YCbCr video files.
http://en.wikipedia.org/wiki/YCbCr

ycbcr.py - is the main class that supports the following formats:

* IYUV
* UYVY
* YV12
* NV12
* YVYU
* YUY2
* 422

Supported operations:

* basic info about a file
* convert between any of the formats above (including correct sub- re-sampling of chroma-data)
* split a file into individual frames
* creates a diff between two files
* PSNR calculations, one value per color-plane
* weighted PSNR
* averaged weighted PSNR
* get luma-data per frame
* SSIM calculation on luma
* convert between 8bpp and 10bpp
* flip left/right, upside/down
* draw frame number in luma-data
* crop
* visualization of PSNR/SSIM using matplotlib
* reduce framerate by throwing away frames

Usage
-----

	$ ./ycbcr.py info foreman_352x288.yuv 352 288 YV12
	$ ./ycbcr.py convert --help
	$ ./ycbcr.py diff --help
	$ ./ycbcr.py split --help
	$ ./ycbcr.py psnr --help
	$ ./ycbcr.py wpsnr --help
	$ ./ycbcr.py wpsnravg --help
	$ ./ycbcr.py ssim --help
	$ ./ycbcr.py 8to10 --help
	$ ./ycbcr.py 10to8 --help
	$ ./ycbcr.py fliplr --help
	$ ./ycbcr.py flipud --help
	$ ./ycbcr.py fnum --help
	$ ./ycbcr.py crop --help
	$ ./ycbcr.py fr --help
	$ ./plot_diff.py foreman_cif_frame_0.yuv foreman_cif_frame_1.yuv 352 288 YV12
	$ ./visual.py psnr_all foreman_cif_frame_0.yuv 352 288 YV12 foreman_cif_frame_1.yuv
Files
-----

* ycbcr.py - main class
* plot_diff.py - matplotlib wrapper around PSNR/SSIM-calculation. Generate nice plots using luma-data.
* verify.py - unittest
* visual.py - matplotlib wrapper around PSNR/SSIM-calculation. Generate nice plots.

Screenshots
-----------

Here's one of the output from visual.py

![psnr](figgis.github.com/yuv-tools/figure_1.png)

![psnr](http://figgis.github.io/yuv-tools/figure_1.png)

