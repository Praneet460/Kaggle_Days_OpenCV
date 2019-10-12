# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

import cv2
import numpy as np

def find_lines(threshold:object, direction = "horizontal", regions = None, linescale = 55) -> object:
    """
    Find the horizontal and vertical lines on the threshold image obtained from `adaptive_threshold` function.

    Parameter
    ---------
    threshold: object
      numpy.ndarray representing the threshold image
    direction: str, optional (default: horizontal)
      Specify whether to find vertical or horizontal lines.
    regions: list, optional (default: None)
      Specify the regions in a list of tuples where you have to find out the lines. The format is [(x1, y1, x2, y2)]
      where, (x1, y1) -> left-top and (x2, y2) -> right-bottom in image coordinate space.
    linescale: int, optional (default: 65)
      Factor by which the page dimensions will be divided to get smallest length of lines that should be detected.
      The larger this value, smaller the detected lines. Making it too large will lead to text being detected as lines.

    Return
    ------
    threshold_mask: object
      numpy.ndarray representing the threshold image of 'vertical' or 'horizontal' lines
      
    """

    if regions is not None:
        region_mask = np.zeros(threshold.shape)
        for region in regions:
            x1, y1, x2, y2 = region
            region_mask[y1:y2, x1:x2] = 1
        threshold = np.multiply(threshold, region_mask)

    if direction == "vertical":
        
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, linescale))
    elif direction == "horizontal":
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (linescale, 1))
    elif direction is None:
        raise ValueError("Specify directions either 'vertical' or 'horizontal'")

    threshold_mask = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, el)

    return threshold_mask

if __name__ == '__main__':
    from thresholding import adaptive_threshold
    adap_threshold = adaptive_threshold('./data/acord.jpeg')
    vertical_mask = find_lines(adap_threshold, direction = 'vertical')
    cv2.imwrite('./output_imgs/vertical_mask.jpeg', vertical_mask)
    horizontal_mask = find_lines(adap_threshold, direction = 'horizontal')
    cv2.imwrite('./output_imgs/horizontal_mask.jpeg', horizontal_mask)

