#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:57:05 2018

@author: luisgonzales
"""

from scipy.ndimage import convolve
import scipy.stats as st


def gkern(kernlen=3, nsig=1):
  '''
  Returns a 2D Gaussian kernel arrayof unit magnitude.
  '''

  interval = (2*nsig+1.)/(kernlen)
  x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw/kernel_raw.sum()
  return kernel


def Gradients(imGray):
  '''
  Obtain Gx and Gy gradients. Gx and Gy are in [-765, 765], as defined by the
  Sobel kernel, but are later squared ([-585225, 585225]). As a result, the
  data type throughout must be int32.
  
  Input:
    imGray: (np.array of np.uint8) Grayscale image
        
  Outputs:
    Gx:     (np.array of np.int32) Convolution of `imGray` and Hx
    Gy:     (np.array of np.int32) Convolution of `imGray` and Hy
  '''
    
  # Convert `imGray` to np.int32 for kernel convolution
  imageFloat = imGray.astype(np.int32)                 # performs inherent copy
  
  # Define kernel and perform convolution
  Hx = np.array([ [-1, 0, 1], 
                  [-2, 0, 2], 
                  [-1, 0, 1] ])
  Gx = convolve(imageFloat, Hx)                          # np.array of np.int32
  Gy = convolve(imageFloat, Hx.T)                        # np.array of np.int32
  
  return (Gx, Gy)


def NonMaxSuppression(mag, theta, thresh):
  '''
  Apply non-max suppression on neighboring magnitudes in `mag` in the direction
  of the gradient, given by `theta`. Values in `mag` less than `thresh`
  (relative to [0,255]) are ignored.
  
  Inputs:
    mag:      (np.array of np.float64) Magnitude of gradient given by
              sqrt(Gx^2 + Gy^2)
    theta:    (np.array of np.float64) Direction of gradient given by
              atan2(-Gy, Gx)
    thresh:   (int) Threshold (in [0,255] scale) below which pixels are ignored
              (kept black)
        
  Output:
    imNonMax: (np.array of np.uint8) Image with non-max suppression applied
  '''
  
  # Map relative (0-255) `thresh` to threshold on `mag` scale and initialize
  # `imNonMax`
  magThresh = thresh * np.max(mag) / 255
  imNonMax  = np.zeros(mag.shape, dtype=np.uint8)
  
  # Map range of atan2 ([-pi, pi]) to nine unique bins
  # i.e., (-pi, -3pi/4, -pi/2, ..., 3pi/4, pi) -> (-4, -3, -2, ..., 3, 4)
  thetaInt = np.round(theta * 4 / np.pi)
  
  # Step through image indices where `mag` is above `magThresh`, apply non-max
  # suppression, and set corresponding index of `imNonMax` high
  (yLim, xLim) = mag.shape
  valCoords = np.nonzero(mag > magThresh)
  for k in range(0, len(valCoords[0])):
    y = valCoords[0][k]
    x = valCoords[1][k]
          
    # Define dx and dy, considering image edges
    dx = 1 if x < (xLim-1) else 0
    dy = 1 if y < (yLim-1) else 0
      
    # Define Booleans
    vertEdge = thetaInt[y,x] ==  0 or thetaInt[y,x] == -4 or thetaInt[y,x] == 4
    horzEdge = thetaInt[y,x] == -2 or thetaInt[y,x] == 2
    diagEdge = thetaInt[y,x] ==  1 or thetaInt[y,x] == -3
    
    # Apply non-max suppression
    if vertEdge and mag[y,x] > mag[y,x-dx] and mag[y,x] > mag[y,x+dx]:
      imNonMax[y,x] = 255
    elif horzEdge and mag[y,x] > mag[y-dy,x] and mag[y,x] > mag[y+dy,x]:
      imNonMax[y,x] = 255
    elif diagEdge and mag[y,x] > mag[y-dy,x+dx] and mag[y,x] > mag[y+dy,x-dx]:
      imNonMax[y,x] = 255
    elif mag[y,x] > mag[y-dy,x-dx] and mag[y,x] > mag[y+dy,x+dx]:
      imNonMax[y,x] = 255

  return imNonMax


def EdgeDet(imGray, thresh=100):
  '''
  Perform edge detection on grayscale image `imGray`. The processing begins
  with Gaussian blurring then obtains discrete derivatives in the x and y
  directions, Gx and Gy. Gx and Gy are then used for non-max suppression with
  an added lower threshold of `thresh`. Overall, the algorithm is similar to
  the popular Canny edge detector but does not apply hysteresis thresholding.
  
  Inputs:
    imGray:   (np.array of np.uint8) Grayscale image
    thresh:   (int) Threshold (in [0,255] scale) below which pixels are ignored
              (kept black)        

  Output:
    imNonMax: (np.array of np.int32) Convolution of `imGray` and Hx
  '''
    
  # Implement Gaussian blurring
  kernel     = gkern()  
  imGrayBlur = convolve(imGray, kernel)                  # np.array of np.uint8
  
  # Obtain Gx and Gy gradients
  Gx, Gy = Gradients(imGrayBlur)                         # np.array of np.int32
  mag    = np.sqrt(Gx**2 + Gy**2)
  theta  = np.arctan2(-Gy, Gx)
  
  # Perform non-max suppression
  return NonMaxSuppression(mag, theta, thresh)           # np.array of np.uint8
