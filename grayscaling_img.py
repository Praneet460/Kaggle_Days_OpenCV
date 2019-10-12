# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

import cv2

def grayscale(image):
    """
    Applies the Grayscale transform. 
    This will return an image with only one color channel. 

    Parameter
    ---------
    image: numpy.narray
      numpy.ndarray of the image

    Return
    ------
    numpy.ndarray of the grayscale image

    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if __name__ == '__main__':
    from reading_img import read_img
    
    img = read_img('./data/acord.jpeg')
    gray_img = grayscale(img)
    cv2.imwrite('./output_imgs/gray_img.jpeg', gray_img)
    print(f"Numpy representation of image\n {gray_img}")
    print(f"Dimension of image = {gray_img.ndim}")
    print(f"Shape of image = {gray_img.shape}")

    img2 = read_img('./data/sample.PNG')
    gray_img2 = grayscale(img2)
    cv2.imwrite('./output_imgs/gray_img2.jpeg', gray_img2)
    print(f"Numpy representation of image\n {gray_img2}")
    # print(f"Dimension of image = {gray_img2.ndim}")
    # print(f"Shape of image = {gray_img2.shape}")


