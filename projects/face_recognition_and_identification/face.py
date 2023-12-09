import numpy as np
import cv2
import pickle
import os


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('C:\\Users\\aky\\Documents\\Programs\\computer-vision\\data\\haarscascade\\haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\aky\\Documents\\Programs\\computer-vision\\data\\haarscascade\\haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:\\Users\\aky\\Documents\\Programs\\computer-vision\\projects\\face_recognition_and_identification\\trainer.yml")


file_path = os.path.join(os.getcwd(), "projects", "face_recognition_and_identification", "labels.pkl")

labels = {}
try:
    with open(file_path, 'rb') as f: #, encoding='utf-8'
        #data = pickle.dumps(labels_ids, pickle.HIGHEST_PROTOCOL)
        labels = pickle.load(f)
        labels = {v:k for k,v in labels.items()}
        #f.close()
except OSError as e:
    print(f"OS error: {e}")


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

        id_, conf = recognizer.predict(roi_gray)
        if  conf>=4 and conf <= 85: 
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

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