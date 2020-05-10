import os
import numpy as np
import pandas as pd
import sys
import face_recognition
import cv2


def get_encoding(folder):
    images = os.listdir(folder)
    encoding = []

    for image in images:
        frame = cv2.imread(os.path.join(folder,image))
        encode = face_recognition.face_encodings(frame)
        if len(encode) > 0:
            encoding.append(encode)

    encoding = np.vstack(encoding)
    encoding = np.average(encoding, axis = 0)
    return encoding

folders = os.listdir('./data')
encoding_dict = {}
for folder in folders:
    encode = get_encoding(os.path.join('data', folder))
    encoding_dict[folder] = encode

df = pd.DataFrame(encoding_dict)
df.to_csv('face_encodings.csv', index=True)
