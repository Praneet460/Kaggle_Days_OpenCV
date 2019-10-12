# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

import cv2
import numpy as np
from sorting_contours import sort_contours

def find_contours(vertical_mask:object, horizontal_mask:object, hierarchy_method = cv2.RETR_EXTERNAL):
    """
    Combining the vertical and horizontal lines and finding contours

    Parameters
    -----------
    vertical_mask: object
      numpy.ndarray representing the threshold image of 'vertical' lines
    horizontal_mask: object
      numpy.ndarray representing the threshold image of 'horizontal' lines
    hierarchy_method: int, optional (default: cv2.RETR_EXTERNAL)
      Type of representation of the relationship between the contours hierarchy.
      Some of the available contour retrieval mode by OpenCV are `RETR_LIST`, `RETR_EXTERNAL`, `RETR_CCOMP`, or `RETR_TREE`.
    
    Return
    ------
    mask: object
      numpy.ndarray representing the threshold image of horizontal and vertical lines
    contours: numpy.array
      numpy array of (x, y) coordinates of boundary points of the object

    """

    alpha = 0.5
    beta = 1 - alpha
    mask = cv2.addWeighted(horizontal_mask, alpha, vertical_mask, beta, 0)
    _ , mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) # increase the white color intensity
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    """
    Before finding contours, apply threshold or canny edge detection.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), hierarchy_method, cv2.CHAIN_APPROX_SIMPLE)

    contours, _ = sort_contours(contours = contours, method = "left-to-right")
    contours, _ = sort_contours(contours = contours, method = "top-to-bottom")

    return mask, contours

if __name__ == '__main__':
    from thresholding import adaptive_threshold
    from finding_lines import find_lines
    adap_threshold = adaptive_threshold('./data/acord.jpeg')
    vertical_mask = find_lines(adap_threshold, direction = 'vertical')
    horizontal_mask = find_lines(adap_threshold, direction = 'horizontal')

    mask, cnts = find_contours(vertical_mask, horizontal_mask)
    cv2.imwrite('./output_imgs/contours.jpeg', mask)
    