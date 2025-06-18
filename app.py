from flask import Flask, render_template, Response, request, redirect, url_for
import cv2, os, pickle, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import csv
import threading
from win32com.client import Dispatch

app = Flask(__name__)

# === Configuration ===
DATA_DIR = 'C:/Users/mohar/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/'
FACE_DATA_PATH = os.path.join(DATA_DIR, 'face_data_4.pkl')
NAMES_PATH = os.path.join(DATA_DIR, 'names_4.pkl')
CASCADE_PATH = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')
ATTENDANCE_DIR = 'Attendance_4'
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# === Globals ===
face_cap = cv2.CascadeClassifier(CASCADE_PATH)
face_data, names, name = [], [], None
i = 0
marked_names = set()

# === Voice Utility ===
def speak(text):
    def _speak():
        speaker = Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
    threading.Thread(target=_speak).start()  # non-blocking

# === Routes ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global name, face_data, i
    if request.method == 'POST':
        name = request.form['username']
        face_data = []
        i = 0
        return redirect(url_for('video_feed_register'))
    return render_template('register.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/video_feed_register')
def video_feed_register():
    return Response(capture_frames_register(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_recognize')
def video_feed_recognize():
    return Response(capture_frames_recognize(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === Video Frame Handlers ===

def capture_frames_register():
    global face_data, i, name
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cap.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            resized = cv2.resize(crop, (50, 50))
            if len(face_data) < 100 and i % 10 == 0:
                face_data.append(resized)
            i += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, str(len(face_data)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if len(face_data) == 100:
            cap.release()
            save_data()
            speak(f"{name}, you have been registered successfully.")
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def capture_frames_recognize():
    with open(NAMES_PATH, 'rb') as f:
        labels = pickle.load(f)
    with open(FACE_DATA_PATH, 'rb') as f:
        faces = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)

    cap = cv2.VideoCapture(0)
    global marked_names

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cap.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in detected:
            crop = frame[y:y+h, x:x+w]
            resized = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1)
            pred = knn.predict(resized)
            user = str(pred[0])
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, user, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            if user not in marked_names:
                ts = datetime.now()
                date = ts.strftime("%d-%m-%Y")
                time_str = ts.strftime("%H:%M:%S")
                file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date}.csv")
                write_header = not os.path.exists(file)

                with open(file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(['Name', 'Time'])
                    writer.writerow([user, time_str])
                
                marked_names.add(user)
                speak(f"{user}, your attendance is recorded. You may go now.")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# === Save Function ===
def save_data():
    face_np = np.asarray(face_data).reshape(100, -1)

    if os.path.exists(FACE_DATA_PATH):
        with open(FACE_DATA_PATH, 'rb') as f:
            old_faces = pickle.load(f)
        face_np = np.append(old_faces, face_np, axis=0)

    with open(FACE_DATA_PATH, 'wb') as f:
        pickle.dump(face_np, f)

    if os.path.exists(NAMES_PATH):
        with open(NAMES_PATH, 'rb') as f:
            old_names = pickle.load(f)
        names = old_names + [name] * 100
    else:
        names = [name] * 100

    with open(NAMES_PATH, 'wb') as f:
        pickle.dump(names, f)

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
