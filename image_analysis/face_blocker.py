import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('C:\\Users\\randu\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\cv2\data\haarcascade_frontalface_default.xml')

def block_face(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(mask,(x,y-int(h*0.2)),(x+int(w*2),int(y+h*1.4)),0,-1)
    return mask
