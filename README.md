YCbCr
=====

Tools for various operations on raw YCbCr video files.

Supported formats:

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
* PSNR calculations

Usage
-----

	$ ./ycbcr.py info foreman_352x288.yuv 352 288 YV12
	$ ./ycbcr.py convert --help
	$ ./ycbcr.py diff --help
	$ ./ycbcr.py split --help
	$ ./ycbcr.py psnr --help
