import cv2 as cv
import sys


# Reading Images
# img = cv.imread('Photos/cat_large.jpg')
#img = cv.imread('../../data/photos/cat.jpg')

img = cv.imread('C:\\Users\\aky\\Documents\\Programs\\computer-vision\\data\\photos\\cat_large.jpg')
# img = cv.imread(cv.samples.findFile('cat.jpg'))

if img is None:
    sys.exit("Could not read the image.")

cv.imshow('Cat',img)
# cv.waitKey(5)
k = cv.waitKey(0)

if k == ord('s'):
    cv.imwrite('cat_copy.png',img)
    cv.destroyAllWindows()

#cv.destroyAllWindows()


# #capture = cv.VideoCapture(0)
# capture = cv.VideoCapture('Video/dog.mp4')

# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video',frame)

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()


