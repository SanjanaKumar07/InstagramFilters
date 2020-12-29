# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:16:33 2020

@author: vetur
"""

import dlib
import numpy as np
import cv2
from math import hypot

cap = cv2.VideoCapture(0)
img1 = cv2.imread("leftHorn.png")
img2 = cv2.imread("rightHorn.png")
img3 = cv2.imread("new.png")

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
        faceWidth = x2 - x1
        faceHeight = y2 - y1
        landmarks = predictor(gray, face)
        leftEyebrow = (landmarks.part(21).x,landmarks.part(21).y)
        rightEyebrow = (landmarks.part(22).x,landmarks.part(22).y)
        leftHeadLeft = (landmarks.part(0).x,landmarks.part(0).y)
        leftHeadRight = (landmarks.part(20).x,landmarks.part(20).y)
        leftLip = (landmarks.part(48).x,landmarks.part(48).y)
        rightLip = (landmarks.part(54).x,landmarks.part(54).y)
        
        hornWidth = int(hypot(leftHeadLeft[0] - leftHeadRight[0],
                              leftHeadLeft[1] - leftHeadRight[1]))
        hornHeight = int(1.5 * hornWidth)

        haloWidth = int((x2 - x1))
        haloHeight = int(haloWidth * 0.559)
        
        if (rightEyebrow[0] - leftEyebrow[0])/faceWidth < 0.11:
            print("angry")
            
            #left horn : 
            leftHornCenter = (landmarks.part(18).x ,landmarks.part(18).y - 20)
            leftTopLeft = (int(leftHornCenter[0] - hornWidth / 2),int(leftHornCenter[1] - hornHeight / 2))
            leftBottomRight = (int(leftHornCenter[0] + hornWidth / 2),int(leftHornCenter[1] + hornHeight / 2))
            
            leftHornImage = cv2.resize(img1, (hornWidth,hornHeight))
            leftHornImageGray = cv2.cvtColor(leftHornImage, cv2.COLOR_BGR2GRAY)
            _, leftHornMask = cv2.threshold(leftHornImageGray, 25, 255, cv2.THRESH_BINARY_INV)
            
            leftHornArea = frame[leftTopLeft[1] : leftTopLeft[1] + hornHeight,
                             leftTopLeft[0] : leftTopLeft[0] + hornWidth]
            
            leftHornAreaNoHorn = cv2.bitwise_and(leftHornArea,leftHornArea, mask = leftHornMask)
            
            finalLeftHorn = cv2.add(leftHornAreaNoHorn,leftHornImage)
            frame[leftTopLeft[1] : leftTopLeft[1] + hornHeight,leftTopLeft[0] : leftTopLeft[0] + hornWidth] = finalLeftHorn
            
            #right horn : 
            
            rightHornCenter = (landmarks.part(25).x ,landmarks.part(25).y - 20)
            rightTopLeft = (int(rightHornCenter[0] - hornWidth / 2),int(rightHornCenter[1] - hornHeight / 2))
            rightBottomRight = (int(rightHornCenter[0] + hornWidth / 2),int(rightHornCenter[1] + hornHeight / 2))
            
            rightHornImage = cv2.resize(img2, (hornWidth,hornHeight))
            rightHornImageGray = cv2.cvtColor(rightHornImage, cv2.COLOR_BGR2GRAY)
            _, rightHornMask = cv2.threshold(rightHornImageGray, 25, 255, cv2.THRESH_BINARY_INV)
            
            rightHornArea = frame[rightTopLeft[1] : rightTopLeft[1] + hornHeight,
                             rightTopLeft[0] : rightTopLeft[0] + hornWidth]
            
            rightHornAreaNoHorn = cv2.bitwise_and(rightHornArea,rightHornArea, mask = rightHornMask)
            
            finalRightHorn = cv2.add(rightHornAreaNoHorn,rightHornImage)
            frame[rightTopLeft[1] : rightTopLeft[1] + hornHeight,rightTopLeft[0] : rightTopLeft[0] + hornWidth] = finalRightHorn
            
            
        elif ((rightLip[0] - leftLip[0])/faceWidth) > 0.43:
            print("smiling")
            haloCenter = (int((x1 + x2) / 2), y1 - 10)
            haloTopLeft = (int(haloCenter[0] - haloWidth/2),int(haloCenter[1] - haloHeight/2))
            haloBottomRight = (int(haloCenter[0] + haloWidth/2),int(haloCenter[1] + haloHeight/2))
            
            haloImage = cv2.resize(img3, (haloWidth,haloHeight))
            haloImageGray = cv2.cvtColor(haloImage, cv2.COLOR_BGR2GRAY)
            _, haloMask = cv2.threshold(haloImageGray, 25, 255, cv2.THRESH_BINARY_INV) 
            
            haloArea = frame[haloTopLeft[1] : haloTopLeft[1] + haloHeight,
                             haloTopLeft[0] : haloTopLeft[0] + haloWidth]
            
            haloAreaNoHalo = cv2.bitwise_and(haloArea,haloArea, mask = haloMask)
            
            finalHalo = cv2.add(haloAreaNoHalo,haloImage)
            
            frame[haloTopLeft[1] : haloTopLeft[1] + haloHeight,haloTopLeft[0] : haloTopLeft[0] + haloWidth] = finalHalo
            
    
    cv2.imshow("frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break