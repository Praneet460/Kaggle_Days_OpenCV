# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

import cv2
import numpy as np

def find_joints(contours:list, vertical_mask:object, horizontal_mask:object) -> dict:
    """
    Extract the joints inside the given contour

    Parameters
    -----------
    contours: numpy.array
      List of contours in the format `[(x, y, w, h)]`
    vertical_mask: object
      numpy.ndarray representing the threshold image of 'vertical' lines
    horizontal_mask: object
      numpy.ndarray representing the threshold image of 'horizontal' lines

    Return
    -------
    table: dict
      Return a dictionay which is having it's key as the contour's coordinates ((x0, y0, x1, y1)) and value will be the list of points of intersection of two lines.
    """
    joints = np.multiply(vertical_mask, horizontal_mask)
    table = {}
    for cnt in contours:
        x, y, w, h = cnt
        roi = joints[y:y+h, x:x+w]
        jc, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if len(jc) <= 4:
            continue
        table[(x, y, x+w, y+h)] = []
        for j in jc:
            jx, jy, jw, jh = cv2.boundingRect(j)
            c1, c2 = x + (2 * jx + jw) // 2, y + (2 * jy + jh) // 2
            table[(x, y, x+w, y+h)].append((c1, c2))
    
    return table

if __name__ == '__main__':
    from thresholding import adaptive_threshold
    from finding_lines import find_lines
    from finding_contours import find_contours
    adap_threshold = adaptive_threshold('./data/acord.jpeg')
    vertical_mask = find_lines(adap_threshold, direction = 'vertical')
    horizontal_mask = find_lines(adap_threshold, direction = 'horizontal')
    mask, cnts = find_contours(vertical_mask, horizontal_mask)

    # finding bounding boxes
    bounding_box_contours = []
    for cnt in cnts:
        x, y, w, h =  cv2.boundingRect(cnt)
        bounding_box_contours.append((x, y, w, h))
    joints_table = find_joints(bounding_box_contours, vertical_mask, horizontal_mask)
    
    print(joints_table)
    