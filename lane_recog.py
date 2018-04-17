#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:28:43 2018

@author: luisgonzales
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:28:43 2018

@author: luisgonzales
"""

# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys
from moviepy.editor import VideoFileClip


def DrawLines(lines, extrap, imShape, yTop, yBtom, xMid):
  '''
  Draw Hough lines atop an otherwise blank image of dimensions imShape (3-D)
  
  Inputs:
    lines:   (np.array) Array containing x,y coordinate end-points of a line
             detected from the Hough transform
    extrap:  (Boolean) Extrapolate drawn lines if True
    imShape: (tuple) Image dimensions (3-D)
    yTop:    (int) y-coordinate associated to top of bounding region
    yBtom:   (int) y-coordinate associated to bottom of bounding region
    xMid:    (int) x-coordinate associated to midpoint of bounding region
  Output:
    imHough: (np.array of np.uint8) Image with lane lines drawn
  '''    
  
  houghColor = (150,150,150)
  imHough = np.zeros(imShape, np.uint8)             # 3D np.ndarray of np.uint8
  
  if extrap:
    xRight = np.array([], dtype=np.uint16)
    yRight = np.array([], dtype=np.uint16)
    xLeft  = np.array([], dtype=np.uint16)
    yLeft  = np.array([], dtype=np.uint16)  
  
  # Separate lines associated with left and right lanes depending on slope
  for line in lines:
    for x1,y1,x2,y2 in line:
      #cv2.line(imHough, (x1,y1), (x2,y2), houghColor, 1) # viz all Hough lines
      slope = (y2-y1)/(x2-x1)
      if slope >= 0 and x1 > xMid:
        if not extrap: cv2.line(imHough, (x1,y1), (x2,y2), houghColor, 5)
        else:
          xRight = np.concatenate( (xRight, [x1], [x2]) )
          yRight = np.concatenate( (yRight, [y1], [y2]) )
      elif slope <= -0.1 and x1 < xMid:
        if not extrap: cv2.line(imHough, (x1,y1), (x2,y2), houghColor, 5)
        else:
          xLeft = np.concatenate( (xLeft, [x1], [x2]) )
          yLeft = np.concatenate( (yLeft, [y1], [y2]) )
    
  # If extrapolating, obtain x-coordinates associated with yTop and yBtom
  if extrap:
    m, b   = np.polyfit(xRight, yRight, 1)  
    xTopR  = int( (yTop - b)/m )
    xBtomR = int( (yBtom - b)/m )
  
    m, b   = np.polyfit(xLeft, yLeft, 1)  
    xTopL  = int( (yTop - b)/m )
    xBtomL = int( (yBtom - b)/m )
  
    cv2.line(imHough, (xTopR, yTop), (xBtomR, yBtom), houghColor, 18)
    cv2.line(imHough, (xTopL, yTop), (xBtomL, yBtom), houghColor, 18)
  
  return imHough


def ProcessImage(im):
  '''
  Perform lane recognition in a specified region of image `im`.
  
  Inputs:
    im:        (np.array of np.uint8) Color image to perform lane recognition on
  Output:
    laneRecog: (np.array of np.uint8) Color image output with lanes drawn
               atop input `im`.
  '''  
  
  extrapLines = True
  
  # Blur image and convert to HSL colorspace
  imBlur = cv2.GaussianBlur(im, (9,9), 0)
  hls = cv2.cvtColor(imBlur, cv2.COLOR_RGB2HLS)
  
  # Define lower and upper threshold for what is considered yellow and white
  lowerY = np.array([0,  0, 110])
  upperY = np.array([60, 255, 255])
  maskY  = cv2.inRange(hls, lowerY, upperY)
  lowerW = np.array([0,  200, 0])
  upperW = np.array([179, 255, 255])
  maskW  = cv2.inRange(hls, lowerW, upperW)  
  maskYW = cv2.bitwise_or(maskY, maskW)

  # Set pixels that can be considered yellow or white to pure white
  imBlur[maskYW>200] = 255
  imBlur[maskYW>200] = 255
  imBlur[maskYW>200] = 255

  # Perform edge detection
  lowThresh  = 200
  highThresh = 395
  imGray = cv2.cvtColor(imBlur, cv2.COLOR_RGB2GRAY) # 2D np.ndarray of np.uint8
  imEdgeDet = cv2.Canny(imGray, lowThresh, highThresh)

  # Only retain edge-detected pixels within a specified region for Hough
  mask      = np.zeros_like(imEdgeDet)                 # np.ndarray of np.uint8  
  topLeft   = (int(430/960*im.shape[1]), int(335/540*im.shape[0]))       # x, y
  topRight  = (int(555/960*im.shape[1]), int(335/540*im.shape[0]))
  btomLeft  = (int(120/960*im.shape[1]), int(513/540*im.shape[0]))
  btomRight = (int(875/960*im.shape[1]), int(513/540*im.shape[0]))
  vertices  = np.array([[topLeft,topRight, btomRight, btomLeft]], dtype=np.int32)
  cv2.fillPoly(mask, vertices, 255)
  imEdgeDetMask = cv2.bitwise_and(imEdgeDet, mask)     # np.ndarray of np.uint8

  # Perform Hough transform on masked `imEdgeDetMask`
  rhoRes    = 1
  thetaRes  = 1*np.pi/180
  threshold = 8                                      # Num votes in accumulator
  minLength = 4
  maxGap    = 1
  lines     = cv2.HoughLinesP(imEdgeDetMask, rhoRes, thetaRes, threshold,
                              np.array([]), minLength, maxGap)
  
  # Obtain image with Hough lines drawn (`imHough`) and overlay on input `im`
  imHough   = DrawLines(lines, extrapLines, im.shape, int(1.05*topLeft[1]),
                        int(0.95*btomLeft[1]), int((topLeft[0]+topRight[0])/2))
  laneRecog = cv2.addWeighted(im, 1, imHough, 1, 0)

  # Add bounding box to `imHough` and overlay on `imEdgeDet`
  #boundColor = (255,255,0)
  #for k in range(vertices.shape[1]):
  #  l = (k+1) % (vertices.shape[1])
  #  x1, y1 = vertices[0, k, 0], vertices[0, k, 1]
  #  x2, y2 = vertices[0, l, 0], vertices[0, l, 1]
  #  cv2.line(imHough, (x1,y1), (x2,y2), boundColor, 2)
  #imEdgeDet_3    = np.dstack((imEdgeDet, imEdgeDet, imEdgeDet)) 
  #imEdgeDetHough = cv2.addWeighted(imEdgeDet_3, 1, imHough, 1, 0)
  
  return laneRecog


# Parse inputs
imPath   = sys.argv[1]                                  # e.g., input/image.jpg
filePath = imPath.split('.')[0]                         # e.g., input/image
file     = filePath.split('/')[1]                       # e.g., image 
ext      = imPath.split('.')[1]                         # e.g., jpg

# Perform routine on image or video
if ext == 'jpg':
  im = mpimg.imread(imPath)                         # 3D np.ndarray of np.uint8
  imLaneRecog = ProcessImage(im)
  plt.imsave('output/' + file + '.jpg', imLaneRecog)

elif ext == 'mp4':
  vidClip    = VideoFileClip(imPath)
  vidResult  = vidClip.fl_image(ProcessImage)
  vidResult.write_videofile('output/' + file + '.mp4', audio=False)
