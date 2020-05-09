import numpy as np
import face_recognition
import cv2
from generate_encoding import *

cap = cv2.VideoCapture(0)

folders = ['Kamlesh', 'Manoj', 'Shubham']

encoding_dict = {}
for folder in folders:
    encode = getEncoding(folder)
    encoding_dict[folder] = encode

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if len(face_encoding) <= 0:
            continue
        #color for blue pass (255, 0, 0)
        frame = cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 255, 255), thickness = 2)
        result = face_recognition.compare_faces(list(encoding_dict.values()), face_encoding)
        name = (folders[np.argmax(result)])
        cv2.putText(frame, name, (bottom, left), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()