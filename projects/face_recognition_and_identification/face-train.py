import cv2 as cv
import os
from PIL import Image
import numpy as np

import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "..\\..\\data\\labeled_faces")
#print(image_dir)

face_cascade = cv.CascadeClassifier('C:\\Users\\aky\\Documents\\Programs\\computer-vision\\data\\haarscascade\haarcascade_frontalface_default.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

current_id = 0
labels_ids = {}

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "").lower() # root = os.path.dirname(path)
            # print(label, path)
            if not label in labels_ids:
                labels_ids[label] = current_id
                current_id += 1
            
            id_ = labels_ids[label]
            #print(labels_ids)
            #x_train.append(path) # some number
            #y_labels.append(label) # verify this iamge, turn into a NUMPY array. GRAY
            pil_image = Image.open(path).convert("L") # L = grayscale
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


            
# print(y_labels)
# print(x_train)
file_path = os.path.join(os.getcwd(), "projects", "face_recognition_and_identification", "labels.pkl")
print(file_path)
try:
    with open(file_path, 'wb') as f:
        pickle.dump(labels_ids, f)
except OSError as e:
    print(f"OS error: {e}")
    
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")