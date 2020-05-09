import numpy as np
import face_recognition
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    face_location = face_recognition.face_locations(frame)
    for (top, right, bottom, left) in face_location:
        #color for blue pass (255, 0, 0)
        frame = cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 255, 255), thickness = 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()