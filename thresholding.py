# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

import cv2
import numpy as np
from gaussian_bluring import gaussian_blur
from grayscaling_img import grayscale


def adaptive_threshold(image_path:str, blocksize = 11, c = -2) -> object:
    """
    Threshold an image using OpenCV's adaptive threshold

    Parameters
    ----------
    image_path: str
      Path to image file
    blocksize: int, optional (default: 11)
      Size of a pixel neighbourhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    c: int, optional (default: -2)
      Constant substracted from the mean or weighted mean.
      Normally, it is positive but may be zero or negative as well.
  

    Return
    ------
    threshold: object
      numpy.ndarray representing the threshold image

    """

    read_image = cv2.imread(image_path) # read an image
    
    if read_image.ndim == 3:
      grayscaled = grayscale(read_image) # convert to gray scale
    else:
      grayscaled = read_image

    gaussian_blured = gaussian_blur(grayscaled, 5)

    """
    Use adaptive thresholding technique when image has different lightning conditions in different areas.
    This method calculates different threshold values for different areas of the same image.

    More Details
    ------------
    `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`: threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
    """

    threshold = cv2.adaptiveThreshold(np.invert(gaussian_blured), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        cv2.THRESH_BINARY, blocksize, c)

    return threshold

def binary_threshold(image_path:str):
    
    read_image = cv2.imread(image_path) # read an image
    
    if read_image.ndim == 3:
      grayscaled = grayscale(read_image) # convert to gray scale
    grayscaled = read_image

    _ , binary_threshold = cv2.threshold(grayscaled,127,255,cv2.THRESH_BINARY)

    return binary_threshold

if __name__ == '__main__':
    binary_img  = binary_threshold(image_path='./data/sample.PNG')
    cv2.imwrite('./output_imgs/binary_img2.jpeg', binary_img)

    adaptive_img = adaptive_threshold(image_path='./data/sample.PNG')
    cv2.imwrite('./output_imgs/adaptive_img2.jpeg', adaptive_img)





