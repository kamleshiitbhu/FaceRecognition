import numpy as np
import pandas as pd
import os
import face_recognition
import cv2

encoding_dict = dict(pd.read_csv('face_encodings.csv'))
folders = list(encoding_dict.keys())

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        continue
    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    # Our operations on the frame come here
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(small_frame, number_of_times_to_upsample = 3)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if len(face_encoding) <= 0:
            continue
        #color for blue pass (255, 0, 0)
        frame = cv2.rectangle(frame, (left * 10, top * 10), (right * 10, bottom * 10), color=(255, 255, 255), thickness = 1)
        result = face_recognition.compare_faces(list(encoding_dict.values()), face_encoding, tolerance = 0.6)
        name = (folders[np.argmax(result)])
        cv2.putText(frame, name, (left * 10 + 6, bottom * 10 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()