import cv2
import numpy as np
import sys
import time


if len(sys.argv)>1:
    inputImage = cv2.imread(sys.argv[1])
else:
    inputImage = cv2.imread("qr1.png") 

# display barcode and qr code location
def display(im, bbox):
    None

