import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('C:\\Users\\aky\\Documents\\Programs\\computer-vision\\data\\haarscascade\haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\aky\\Documents\\Programs\\computer-vision\\data\\haarscascade\\haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Making a frame for the face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        # Region Of Interest, we are taking the face and converting it to gray to look for the eye
        roi_gray = gray[y:y+h, x:x+w] # ycord_start, ycord_end
        roi_color = frame[y:y+h, x:x+w] 

        
        # Detecting the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        # Making a frame for the eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3) 


    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When  everyuthing done, release the capture
cap.release()
cv2.destroyAllWindows()