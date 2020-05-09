import os
import sys
import pandas as pd
import numpy as np
import face_recognition
import cv2

name = input('Enter Your Name: ')
if os.path.exists('data') == False:
    os.mkdir('data')
name_exists = os.path.exists(os.path.join('data',name))
while name_exists == True:
    print('This name already exists\n')
    name = input('Enter Any Other Name: ')
    name_exists = os.path.exists(os.path.join('data',name))
os.mkdir(os.path.join('data',name))
name_dir = os.path.join('data',name)
count = 0

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    face_location = face_recognition.face_locations(small_frame)
    face_detected = False
    for (top, right, bottom, left) in face_location:
        face_detected = True
        small_frame = small_frame[top :  bottom, left : right]     

    # Display the resulting frame
    cv2.imshow('frame', cv2.resize(small_frame, (0, 0), fx=10, fy=10))
    if face_detected:
        file_name = '{}_{}.jpg'.format(name, count)
        count += 1
        cv2.imwrite(os.path.join(name_dir, file_name), small_frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()