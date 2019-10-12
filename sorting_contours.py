# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

import cv2

def sort_contours(contours:list, method:str):
    """
    Sorting the list of contours in the specified method.

    Parameters
    ----------
    contours: list
       List of contours found on threshold image
    method: str
       Method of sorting the list of contours.
       Available options are 'left-to-right', 'right-to-left', 'top-to-bottom', and 'bottom-to-top'

    Return
    -------
    contours_list: object
       Sorted list of contours
    bounding_boxes: object
       Sorted bounding boxes
       
    """

    reverse = False
    i = 0
    
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(cnt) for cnt in contours]

    contours_list, bounding_boxes = zip(*sorted(zip(contours, boundingBoxes), key = lambda b:b[1][i], reverse = reverse))

    return contours_list, bounding_boxes