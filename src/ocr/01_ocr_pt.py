import cv2
import PIL
import pytesseract

from PIL import Image

PATH = "C:\\Users\\aky\\Documents\\Programs\\computer-vision\\"
im_file = PATH + "data\\pdf\\page_01.jpg"

im = Image.open(im_file)
print(im)
print(im.size)
im.show()

# saving a file
# im.save("C:\\Users\\aky\\Documents\\Programs\\computer-vision\\src\\ocr\\page01.png")