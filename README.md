YCbCr
=====

Tools for various operations on raw YCbCr video files.

Supported formats:

* IYUV
* UYVY
* YV12
* YVYU

Supported operations:

* basic info about a file
* convert between any of the formats above (including correct subsampling of chroma-data)
* split a file into individual frames
* creates a diff between two files

Usage
-----

	$ ./ycbcr.py info foreman_352x288.yuv 352 288 YV12
	$ ./ycbcr.py convert --help
