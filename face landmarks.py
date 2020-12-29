# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:53:23 2020

@author: vetur
"""

import dlib
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert the video into gray for faster processing
    
    faces = detector(gray) #detect faces in the gray frame 
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        landmarks = predictor(gray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x,y), 5, (0,0,255), -1)
    
    
    cv2.imshow("frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break