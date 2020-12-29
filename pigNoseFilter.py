# -*- coding: utf-8 -*-
"""
Created on Tue May 19 00:09:22 2020

@author: vetur
"""
import dlib
import numpy as np
import cv2
from math import hypot

cap = cv2.VideoCapture(0)
noseImage = cv2.imread("pigNose2.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        topNose = (landmarks.part(29).x,landmarks.part(29).y)
        #cv2.circle(frame, topNose, 5, (0,0,255), -1)
        leftNose = (landmarks.part(31).x,landmarks.part(31).y)
        rightNose = (landmarks.part(35).x,landmarks.part(35).y)
        centerNose = (landmarks.part(30).x,landmarks.part(30).y)
        noseWidth = int(hypot(leftNose[0] - rightNose[0],
                          leftNose[1] - rightNose[1]) * 1.7)
        noseHeight = noseWidth 
        
        topLeft = (int(centerNose[0]-noseWidth/2),int(centerNose[1]-noseHeight/2))
        bttomRight = (int(centerNose[0]+noseWidth/2),int(centerNose[1]+noseHeight/2))
        
        pigImage = cv2.resize(noseImage, (noseWidth,noseHeight))
        pigImageGray = cv2.cvtColor(pigImage, cv2.COLOR_BGR2GRAY)
        _, noseMask = cv2.threshold(pigImageGray, 25, 255, cv2.THRESH_BINARY_INV)
        
        noseArea = frame[topLeft[1] : topLeft[1] + noseHeight,
                         topLeft[0] : topLeft[0] + noseWidth]  
        noseAreaNoNose = cv2.bitwise_and(noseArea,noseArea, mask = noseMask)
        finalNose = cv2.add(noseAreaNoNose,pigImage)
        
        frame[topLeft[1] : topLeft[1] + noseHeight,
                         topLeft[0] : topLeft[0] + noseWidth]  = finalNose
        
        cv2.imshow("nose area", noseArea)
        cv2.imshow("pig nose",pigImage)
        cv2.imshow("final nose",finalNose)             
        
        
    cv2.imshow("frame", frame)
        
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
    
