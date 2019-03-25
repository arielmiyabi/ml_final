import os
import cv2
from glob import glob
import numpy as np

data = glob("./data/ShoeV2_F/train_data/*")
label = glob("./data/ShoeV2_F/train_label/*")
newdata = "./data/train_blur_data/"
newlabel = "./data/train_blur_label/"
kernel = np.ones((9,9),np.uint8)

os.makedirs(newdata, exist_ok=True)
os.makedirs(newlabel, exist_ok=True)
for filename in data:
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = 255 - img
    dilation = cv2.dilate(img,kernel,iterations = 1)
    blur = cv2.GaussianBlur(dilation,(15,15),0)
    newfilename = newdata + filename.split('/')[-1]
    print(filename.split('/')[-1])
    cv2.imwrite(newfilename, blur)
# for filename in label:
#     img = cv2.imread(filename, cv2.IMREAD_COLOR)
#     dilation = cv2.dilate(img,kernel,iterations = 1)
#     blur = cv2.GaussianBlur(dilation,(15,15),0)
#     newfilename = newlabel + filename.split('/')[-1]
#     print(filename.split('/')[-1])
#     cv2.imwrite(newfilename, blur)
