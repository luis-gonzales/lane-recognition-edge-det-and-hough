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
from src import hough
from src import edge_det
import sys


# Parse inputs
imPath = sys.argv[1]
numInputs     = len(sys.argv) - 1
useOwnEdgeDet = False
useOwnHough   = False

if numInputs > 1 and sys.argv[2] == 'own':
  useOwnEdgeDet = True
elif numInputs > 1 and sys.argv[2] == 'OpenCV':
  useOwnEdgeDet = False
elif numInputs > 1 and (sys.argv[2] != 'own' or sys.argv[2] != 'OpenCV'):
  raise ValueError('Enter proper input for edge detection type')

if numInputs > 2 and sys.argv[3] == 'own':
  useOwnHough = True
elif numInputs > 2 and sys.argv[3] == 'OpenCV':
  useOwnHough = False
elif numInputs > 2 and (sys.argv[3] != 'own' or sys.argv[3] != 'OpenCV'):
  raise ValueError('Enter proper input for Hough type')

# Read in image and convert to grayscale
im     = mpimg.imread(imPath)                       # 3D np.ndarray of np.uint8
imGray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)       # 2D np.ndarray of np.uint8

# Perform edge detection
lowThresh  = 100
highThresh = 250
if useOwnEdgeDet: 
  imEdgeDet = edge_det.EdgeDet(imGray, lowThresh)      # np.ndarray of np.uint8
else:
  imEdgeDet = cv2.Canny(imGray, lowThresh, highThresh) # np.ndarray of np.uint8
plt.imsave('output/edge-det.jpg', imEdgeDet, cmap=plt.cm.gray)

# Only retain edge-detected pixels within a specified region
mask      = np.zeros_like(imEdgeDet)                   # np.ndarray of np.uint8
topLeft   = (400,350)                                  # col (x), row (y)
topRight  = (540,350)
btomLeft  = (100,538)
btomRight = (830,538)
vertices  = np.array([[topLeft,topRight, btomRight, btomLeft]], dtype=np.int32)
cv2.fillPoly(mask, vertices, 255)
imEdgeDetMask = cv2.bitwise_and(imEdgeDet, mask)       # np.ndarray of np.uint8

# Perform Hough transform on masked `imEdgeDetMask`
rhoRes    = 1
thetaRes  = 1*np.pi/180
threshold = 10                                       # Num votes in accumulator
minLength = 10
maxGap    = 5
if useOwnHough: lines = hough.Hough(imEdgeDetMask, rhoRes, thetaRes, 
                                    threshold, np.array([]), minLength, maxGap)
else: lines = cv2.HoughLinesP(imEdgeDetMask, rhoRes, thetaRes, threshold, 
                                               np.array([]), minLength, maxGap)

# Draw Hough lines on blank (color, 3D) image and overlay on `im`
boundColor = (255,255,0)
houghColor = (255,0,0)
imHough    = np.zeros(im.shape, np.uint8)           # 3D np.ndarray of np.uint8
for line in lines:
  for x1,y1,x2,y2 in line:
    cv2.line(imHough, (x1,y1), (x2,y2), houghColor, 10)
laneRecog = cv2.addWeighted(im, 0.95, imHough, 1, 0)
plt.imsave('output/lane-recognition.jpg', laneRecog)

# Add bounding box to `imHough` and overlay on `imEdgeDet`
for k in range(vertices.shape[1]):
  l = (k+1) % (vertices.shape[1])
  x1, y1 = vertices[0, k, 0], vertices[0, k, 1]
  x2, y2 = vertices[0, l, 0], vertices[0, l, 1]
  cv2.line(imHough, (x1,y1), (x2,y2), boundColor, 2)
imEdgeDet_3    = np.dstack((imEdgeDet, imEdgeDet, imEdgeDet)) 
imEdgeDetHough = cv2.addWeighted(imEdgeDet_3, 1, imHough, 1, 0)
plt.imsave('output/edge-det-hough.jpg', imEdgeDetHough)
