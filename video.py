import cv2, time
import numpy as np
a = 1
firstframe = None
#video = cv2.VideoCapture("C:\\Users\\aadiv.DESKTOP-TEH73VP\\OneDrive\\Documents\\Python\\vid2.mp4")

video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("C:\\Users\\aadiv.DESKTOP-TEH73VP\\OneDrive\\Documents\\Python\\haarcascade_frontalface.xml")

while True:
    check, frame = video.read()
    #newframe = cv2.resize(frame, ( int(frame.shape[1]/6),int(frame.shape[0]/6) ) )
    gray = (frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray ,(21,21),0)

    if firstframe is None:
        firstframe = gray
        continue
    cv2.imshow('Capturing',gray)
    deltaframe = cv2.absdiff(firstframe,gray)
    threshdelta = cv2.threshold(deltaframe,30,255,cv2.THRESH_BINARY)[1]
    threshdelta = cv2.dilate(threshdelta,None,iterations=0)
    cnts, _ = cv2.findContours(threshdelta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=5, minNeighbors=10)

    for x,y,w,h in faces:
        gray = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.imshow('frame',frame)
        cv2.imshow('delta',deltaframe)
        cv2.imshow('thresh',threshdelta)

        cv2.imshow('Capturing',gray)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
