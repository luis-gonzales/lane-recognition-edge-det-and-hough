#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 01:26:55 2018

@author: luisgonzales
"""

import copy
from random import randint
import numpy as np

def ColumnInArray(col, array):
  return np.any(np.all(col == array, axis=0))


def getMaxLine(inArray, theta, minLength, maxGap):
  '''
  getMaxLine() considers all white pixels in the corridor defined by `theta`,
  given by `inArray`, and outputs the longest line segment in the corridor.
  Groups of pixels are considered a line if the end points have a length
  greater than `minLength` and there is no gap in between larger than `maxGap`.
  The general approach is to map `inArray` to x',y' coordinates, sort, and
  traverse to find longest line. Lastly, the pairs in the longest line from
  x',y' coordinates are converted back to x,y coordinates before outputting.
  
  Inputs:
    inArray:   (np.array) 2-by-n array where n is the number of white pixels
               in the corridor defined by `theta`. Each column is given by an
               [x;y] pair in x,y coordinates and are assumed to be unsorted
               relative to one another.
    theta:     (float) Angle (radians) from polar coordinates with which the
               corridor was built.
    minLength: (int) Minimum length to be considered a valid line
    maxGap:    (int) Maximum length of gap allowed within longest line
        
  Output:
    outArray:  (np.array) All constituent points making up the longest line in
               x,y coordinates
  '''

  # Define and apply rotation operator to get array in x', y' frame
  O = np.array([ [np.cos(theta), -np.sin(theta)],
                 [np.sin(theta),  np.cos(theta)] ])
  O_T = np.transpose(O)
  inArrayP = O_T.dot(inArray)
  
  # Sort inArrayP to traverse to find segment with maximum length
  idx = np.argsort(inArrayP[1,:])
  #print('idx = ', idx)  
  sortedInArrayP = inArrayP[:, idx]
  #print('sorted_np_prime = ', sorted_np_prime)
  
  # Initialize indexing and length variables
  curTail, curHead, curTrackLen = 0, 0, 0
  maxTail, maxHead, maxTrackLen = 0, 0, 0

  # Traverse through sortedInArray to find length of max length as pointed to
  # by maxTail and maxHead
  for curIdx in range(1,len(idx)):
    distToPrev = sortedInArrayP[1,curIdx] - sortedInArrayP[1,curIdx-1]
    if distToPrev <= maxGap:
      curTrackLen += distToPrev
      curHead = curIdx
    else:
      # Store if appropriate; reset
      if curTrackLen >= minLength and curTrackLen > maxTrackLen:
        maxTail, maxHead, maxTrackLen = curTail, curHead, curTrackLen
      curTail, curHead, curTrackLen = curIdx, curIdx, 0
  if curTrackLen >= minLength and curTrackLen > maxTrackLen:
    maxTail, maxHead, maxTrackLen = curTail, curHead, curTrackLen
  
  # Output empty array if maxTail and maxHead haven't advanced; otherwise,
  # use them to grab appropriate indices of non-sorted array to build outArray
  outArray = np.array([ [],[] ])
  if (not maxTail) and (not maxHead): return outArray
  sortedIdx    = np.arange(maxTail, maxHead+1)
  nonSortedIdx = idx[sortedIdx]
  outArray     = inArray[:, nonSortedIdx]  
  return outArray


def CorridorExtract(whtPixels, rho, theta):
  '''
  `CorridorExtract` considers all white pixels remaining in the image, given
  by `whtPixels`, and returns those that are in a corridor defined by `rho` and
  `theta` as `outArray`. The approach is to use the relation of
  p = x cos(theta) + y sin(theta) for all pairs.
  
  Inputs:
    whtPixels: (np.array) 2-by-n array where n is the number of white pixels
               remaining in the image. Each column is given by an [x;y] pair in
               x,y coordinates and are assumed to be unsorted relative to one
               another.
    rho:       (float) Perpendicular vector in polar coordinates to the
               corridor to be built.
    theta:     (float) Angle (radians) in polar coordinates with which the
               corridor is to be built.
        
  Output:
    outArray:  (np.array) All white pixels within the corridor in x,y
               coordinates
    
  Note: Attempted is build corridor using `while` and `for` loops, and rotation
  operators, but the approach resulted in terribly slow performance.
  '''
  
  # Define corridor width and 1x2 vector  
  corridorWidth = 1
  v = np.array([np.cos(theta), np.sin(theta)])
  
  # Calculate rhos for each x,y pair in `whtPixels`; for rhos that are within
  # +/- `corridorWidth` of `rho`, output associated whtPixels
  rhoVals = v.dot(whtPixels)
  idx = np.abs(rho - rhoVals) < corridorWidth
  outArray = whtPixels[:, idx]

  return outArray


def Hough(image, rhoStep, thetaStep, accumThresh, array, minLength, maxGap):
  '''
  `Hough` performs a variant of the Progressive Probabilistic Hough Transform
  proposed by Matas, et al. Once a given element of the accumulator has enough
  evidence (`accumThresh`) of a line, the algorithm proceeds to collect all the
  remaining points given by the corresponding corridor.
  
  Inputs:
    image:      (np.array) Edge-detected, black-and-white image
    rhoStep:    (int) Resolution of rho values in accumulator
    thetaStep:  (float) Resolution (in radians) of theta in accumulator
    accumThres: (int) Accumulator threshold to build corresponding corridor
    array:      (np.array) Unused input only included here so that input
                arguments are equivalent to those in OpenCV cv2.HoughLinesP
    minLength:  (int) Minimum line length to qualify as line
    maxGap:     (int) Maximum length of gap allowed within a line
     
  Output:
    xyList:     (list) Each entry of the list contains the x,y coordinates of
                the two endpoints of a line found (i.e., [x0, y0, x1, y1])
  ''' 
  
  xyList = []
  yLim, xLim = image.shape
  
  # Define rho and theta arrays
  rho      = np.ceil(np.sqrt(yLim**2 + xLim**2))
  rhoVec   = np.linspace(-rho, rho, int(np.ceil(2*rho/rhoStep)))
  thetaVec = np.linspace(-np.pi/2, np.pi/2, int(np.ceil(np.pi/thetaStep)))
  rhoDelta = -rhoVec[0] + rhoVec[1]
  lenThVec = len(thetaVec)
  
  # Initialize accumulator
  accum = [ np.array([[],[]], dtype=np.int16) \
            for i in range(lenThVec*len(rhoVec)) ]
    
  # Obtain [x;y] coordinates of white pixels
  yIdx, xIdx = np.nonzero(image > 200)
  xyWhite    = np.array((xIdx, yIdx))
  
  # Continue until all white pixels are accounted for
  while xyWhite.shape[1]:
    
    # Draw random pixel from remaining pixels to be considered and reset
    k    = randint(0, xyWhite.shape[1]-1)
    x, y = xyWhite[0,k], xyWhite[1,k]
    foundMax = False
    
    # Update accumulator by stepping through each theta; calculate rho based
    # on current theta and find optimal, corresponding index in rhoVec
    for thetaIdx in range(lenThVec):                    # go through each theta
      rhoVal = x*np.cos(thetaVec[thetaIdx]) + y*np.sin(thetaVec[thetaIdx])
      rhoIdx = int(np.rint((rhoVal + rho)/rhoDelta))
      accum[rhoIdx*lenThVec+thetaIdx] = np.concatenate(( \
                                             accum[rhoIdx*lenThVec+thetaIdx], \
                                             np.array([ [x],[y] ])), axis=1)
      currAccumSz = accum[rhoIdx*lenThVec+thetaIdx].shape[1]
      if (not foundMax) and (currAccumSz >= accumThresh):
        foundMax = True
        rhoMaxIdx, thetaMaxIdx = rhoIdx, thetaIdx
    
    # Remove current pixel from `xyWhite`
    mask   = np.any(xyWhite != xyWhite[:,[k]], axis=0)
    xyWhite = xyWhite[:,mask]
    
    # Check if max value in accumulator leads to enough evidence for corridor
    if foundMax:
      
      # Define rho and theta vals at accumulator max and get pixels along
      # corresponding corridor
      accumMaxRho   = rhoVec[rhoMaxIdx]
      accumMaxTheta = thetaVec[thetaMaxIdx]
      xyCorridor    = CorridorExtract(xyWhite, accumMaxRho, accumMaxTheta)
      
      # Combine points in accumulator max and those from the corridor and get
      # the x,y coordinates of the points that make up the longest line
      xyCombined = np.concatenate((accum[rhoMaxIdx*lenThVec+thetaMaxIdx], \
                                   xyCorridor), axis=1)
      xyLongest  = getMaxLine(xyCombined, accumMaxTheta, minLength, maxGap)
      
      # If there's a valid line for the combined points, proceed
      if xyLongest.shape[1]:
        
        # Remove `xyCorridor` pts that are also in `xyLongest` from `xyWhite`
        # (no need to modify accumulator since these pts would not yet have
        # made it in by defn of corridor)
        for currCorridor in xyCorridor.T:
          currCorridor = currCorridor[np.newaxis, :].T             # col vector
          if ColumnInArray(currCorridor, xyLongest):
            mask    = np.any(xyWhite != currCorridor, axis=0)
            xyWhite = xyWhite[:,mask]
                
        # Unvote points in accumulator max that are part of `xyLongest`
        # (deepcopy due to active modification of accumulator)
        accumMaxCopy = copy.deepcopy(accum[rhoMaxIdx*lenThVec+thetaMaxIdx])
        for accumMaxPair in accumMaxCopy.T:
          accumMaxPair = accumMaxPair[np.newaxis, :].T             # col vector
          if ColumnInArray(accumMaxPair, xyLongest):
            x, y = accumMaxPair[0], accumMaxPair[1]
            
            # Update accumulator by stepping through each theta (sim to above)
            for thetaIdx in range(lenThVec):            # go through each theta
              rhoVal = x*np.cos(thetaVec[thetaIdx]) \
                     + y*np.sin(thetaVec[thetaIdx])
              rhoIdx = int(np.rint((rhoVal + rho)/rhoDelta))
              mask   = np.any(accum[rhoIdx*lenThVec+thetaIdx] \
                           != accumMaxPair, axis=0)
              accum[rhoIdx*lenThVec+thetaIdx] = \
                                        accum[rhoIdx*lenThVec+thetaIdx][:,mask]
              
        # Add head and tail of `xyLongest` to output list, `xyList`
        xyList.append( np.array( [[xyLongest[0,0], xyLongest[1,0], \
                                   xyLongest[0,-1], xyLongest[1,-1]]] ) )

  return xyList
