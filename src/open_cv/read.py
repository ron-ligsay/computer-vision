import cv2 as cv

# Reading Images
"""
#img = cv.imread('Photos/cat_large.jpg')
img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat',img)
cv.waitKey(0)
"""


#capture = cv.VideoCapture(0)
capture = cv.VideoCapture('Video/dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video',frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()


