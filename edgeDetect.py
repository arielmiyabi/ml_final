import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


root = "./ShoeV2_F/ShoeV2_photo/"
dstname = "./ShoeV2_F/ShoeV2_photo_edge/"
for file in os.listdir(root):
    filename = root+file
    print (filename)
    img = cv2.imread(filename,0)
    edges = cv2.Canny(img,100,200)
    filename = dstname + file
    cv2.imwrite(filename, edges)
