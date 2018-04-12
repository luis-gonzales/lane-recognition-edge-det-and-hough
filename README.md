# Traffic Lane Recognition using Edge Detection and the Hough Transform
Traffic lane markings are detected from image or video input. Below is a typical implementation, where the overlaid red lines have been generated by the algorithm described below and in further detail at [www.lrgonzales.com/traffic-lane-recognition](http://www.lrgonzales.com/traffic-lane-recognition).

## Algorithm
The algorithm has three steps, summarized below. Edge detection and the Hough transform are covered in further detail at [www.lrgonzales.com/traffic-lane-recognition](http://www.lrgonzales.com/traffic-lane-recognition).
* Grayscale conversion
* Edge detection
* Hough transform for linear feature identification

The Canny edge detector was chosen for edge detection, particularly because edges in the input image are reduced to single-pixel width lines, a useful feature for the following Hough Transform processing. The Hough transform then evaluates the pixels associated to an edge and determines whether a line (lane marking) exists.

To gain experience with kernel convolution and advanced Hough transform implementations, I implemented my own versions of edge detection and the Hough transform. 

## Usage


## Improvements
The `src/` implementation of the Hough transform could benefit from an accumulator not implemented as a `list`. Doing so would likely result in on-par performance to the OpenCV implementation.

Expecting completion of `README` by April 13, 2018.
