
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Constants
DATA_DIR = "C:/Users/mohar/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/"
FACE_CASCADE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
FACE_DATA_PATH = os.path.join(DATA_DIR, "face_data_4.pkl")
NAMES_PATH = os.path.join(DATA_DIR, "names_4.pkl")
ATTENDANCE_DIR = "Attendance_4"
BACKGROUND_IMAGE_PATH = "Dark Grey and White Minimalist Website Register Desktop Prototype.jpg"

# Ensure attendance directory exists
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Load Haar cascade
face_cap = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Load data
with open(NAMES_PATH, 'rb') as f:
    LABELS = pickle.load(f)

with open(FACE_DATA_PATH, 'rb') as f:
    FACES = pickle.load(f)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imagebackground = cv2.imread(BACKGROUND_IMAGE_PATH)

# Start video capture
video_cap = cv2.VideoCapture(0)
COL_NAMES = ['NAME', 'TIME']

# Track names already marked present
marked_names = set()

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    ts = time.time()
    date_str = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
    time_str = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_4_{date_str}.csv")

    for (x, y, w, h) in faces:
        crop_img = video_data[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        try:
            output = knn.predict(resized_img)
        except Exception as e:
            print("Prediction failed:", e)
            continue

        name = str(output[0])

        # Draw face rectangle and name
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(video_data, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Save attendance on key press later
        if name not in marked_names:
            attendance = [name, time_str]

    # Overlay video on background
    try:
        video_data_resized = cv2.resize(video_data, (457, 480))  # width, height
        image_show = imagebackground.copy()
        image_show[150:150 + 480, 55:55 + 457] = video_data_resized
    except:
        image_show = video_data  # fallback to video only

    cv2.imshow("Video_live", image_show)

    k = cv2.waitKey(1)
    if k == ord("o"):  # Save attendance
        speak(f"{name}, your attendance has been recorded. you can go now")
        if name not in marked_names:
            with open(attendance_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not os.path.isfile(attendance_file):
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            marked_names.add(name)
            print(f"Attendance marked for: {name}")
    elif k == ord("a"):  # Exit
        break

video_cap.release()
cv2.destroyAllWindows()
