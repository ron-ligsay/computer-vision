import glob
import cv2
import pandas as pd
import pathlib



def read_qr_code(filename):
    """Read an image and read the QR code.


    Args:
        filename (string): Path to file
    
    Returns:
        qr (string): Value from QR code
    """
    
    
    try:
        img = cv2.imread(filename)
        detect = cv2.QRCodeDetector()
        value, points, straight_qrcode = detect.detectAndDecode(img)
        return value
    except:
        return


