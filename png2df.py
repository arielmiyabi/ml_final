import numpy as np
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
import os
import cv2


root = "./ShoeV2_F/ShoeV2_photo_edge/"
dstname = "./ShoeV2_F/distanceFiled_mask3_photo/"
for file in os.listdir(root):
    filename = root+file
    print (filename)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(thresh,cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist = (1 - dist)*255
    filename = dstname + file
    cv2.imwrite(filename, dist)
