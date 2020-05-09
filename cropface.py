import os
import sys
import pandas as pd
import numpy as np
import face_recognition
import cv2

name = sys.argv[1]
os.mkdir(name)
name_dir = './' + name + '/'
count = 0

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    face_location = face_recognition.face_locations(frame)
    for (top, right, bottom, left) in face_location:
        frame = frame[top :  bottom, left : right]     

    # Display the resulting frame
    cv2.imshow('frame', frame)
    file_name = '{}_{}.jpg'.format(name, count)
    count += 1
    cv2.imwrite(os.path.join(name_dir, file_name), frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()