# -*- coding: utf-8 -*-

##########################
# AUTHOR : PRANEET NIGAM
##########################

# import third-party library
import cv2

def read_img(img_path:str) -> object:
    """
    Read the image using OpenCV
    """
    img = cv2.imread(img_path)
    return img

if __name__ == '__main__':
    img = read_img(img_path = './data/acord.jpeg')
    print(f"Numpy representation of image\n {img}")
    print(f"Dimension of image = {img.ndim}")
    print(f"Shape of image = {img.shape}")

    img2 = read_img(img_path= './data/sample.PNG')
    # print(f"Numpy representation of image\n {img2}")
    # print(f"Dimension of image = {img2.ndim}")
    # print(f"Shape of image = {img2.shape}")


