# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

import cv2
import numpy as np

def gaussian_blur(img, kernel_size:int):
    """
    Applies a Gaussian Noise kernel.
    This is needed for the canny edge detection to average out 
    anomalous gradients in the image.

    Parameter
    ---------
    img: numpy.ndarray
      Image read in the numpy array format
    kernel_size: int
      kernel size in the `int` format. It's a kind of 
      (kernel_size x kernel_size) Gaussian filter.

    Return
    ------
      numpy.ndarray of the GaussianBlur image.
      
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

if __name__ == '__main__':
  from reading_img import read_img
  img  = read_img(img_path = './data/acord.jpeg')

  
  blur_img  = gaussian_blur(img, kernel_size=5)

  cv2.imwrite('./output_imgs/blur_img.jpeg', blur_img)
