
import cv2
import pickle
import numpy as np
import os

# Constants
DATA_DIR = "C:/Users/mohar/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/"
FACE_CASCADE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
FACE_DATA_PATH = os.path.join(DATA_DIR, "face_data_4.pkl")
NAMES_PATH = os.path.join(DATA_DIR, "names_4.pkl")

# Load Haar Cascade
face_cap = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Start video capture
video_cap = cv2.VideoCapture(0)

face_data = []
i = 0
name = input("Enter your name: ")

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        crop_img = video_data[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(face_data) < 100 and i % 10 == 0:
            face_data.append(resized_img)
        i += 1
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(video_data, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

    cv2.imshow("Video_live", video_data)
    if cv2.waitKey(10) == ord("a") or len(face_data) == 100:
        break

video_cap.release()
cv2.destroyAllWindows()

# Convert to NumPy array
face_data = np.asarray(face_data)
face_data = face_data.reshape(100, -1)

# Handle names.pkl
if not os.path.exists(NAMES_PATH):
    names = [name] * 100
else:
    with open(NAMES_PATH, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100

with open(NAMES_PATH, 'wb') as f:
    pickle.dump(names, f)

# Handle face_data.pkl
if not os.path.exists(FACE_DATA_PATH):
    with open(FACE_DATA_PATH, 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(FACE_DATA_PATH, 'rb') as f:
        existing_face_data = pickle.load(f)
    combined_data = np.append(existing_face_data, face_data, axis=0)
    with open(FACE_DATA_PATH, 'wb') as f:
        pickle.dump(combined_data, f)

