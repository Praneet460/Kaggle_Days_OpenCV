# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

from itertools import groupby
from operator import itemgetter

def find_table(table:dict) -> dict:
    """
    Find the coordinates of the table along with number of rows and columns.

    Parameters
    -----------
    table: dict
      A dictionay of table, having key as contours coordinates and value as the joints.

    Return
    -------
    table_dict: dict
      A dictionay having key as the coordinates of the table in this ((x0, y0, x1, y1)) format, and value as the number of rows and columns in this ((number_of_rows, number_of_columns)) format.
      
    """
    table_dict = {}
    for table_key, table_value in table.items():
        table_value = sorted(table_value, key = lambda x: x[0])
        table_value = sorted(table_value, key = lambda x: x[1])

        lst = [(k, list(list(zip(*g))[0])) for k, g in groupby(table_value, itemgetter(1))]

        output_dict = {}

        for lst0, lst1 in lst:
            lst1 = tuple(lst1)
            if lst1 in output_dict:
                output_dict[lst1].append(lst0)
            else:
                output_dict[lst1] = [lst0]
        
        for key, value in output_dict.items(): # key -> columns and value -> rows
            if len(value) > 3: # minimum 3 rows
                num_of_rows = len(value) - 1
                num_of_columns = len(key) - 1
                x0, y0, x1, y1 = key[0], value[0], key[-1], value[-1]
                table_dict[(x0, y0, x1, y1)] = (num_of_rows, num_of_columns)
    
    return table_dict

if __name__ == '__main__':
    import cv2
    from thresholding import adaptive_threshold
    from finding_lines import find_lines
    from finding_contours import find_contours
    from finding_joints import find_joints
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
    table_dict = find_table(table=joints_table)
    print(table_dict)

        
    