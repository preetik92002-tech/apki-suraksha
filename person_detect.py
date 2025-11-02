from flask import Flask, jsonify
import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime

app = Flask(__name__)

# === Load MobileNet SSD for person detection ===
MODEL_PROTOTXT = "deploy.prototxt"
MODEL_WEIGHTS = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_WEIGHTS)

# === Load known faces ===
KNOWN_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

if os.path.isdir(KNOWN_DIR):
    for file in os.listdir(KNOWN_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_DIR, file)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file)[0])

print(f"✅ Loaded known faces: {known_face_names}")

# === Folder for snapshots ===
os.makedirs("snapshots", exist_ok=True)

# === Detection variables ===
current_count = 0
threshold = 3
last_snapshot_time = 0  # track when last snapshot taken
snapshot_cooldown = 10  # seconds between snapshots

video = cv2.VideoCapture(0)

def detect_people():
    global current_count, last_snapshot_time

    while True:
        ret, frame = video.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        person_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:
                    box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                    sX, sY, eX, eY = box
                    sX, sY, eX, eY = max(0, sX), max(0, sY), min(w, eX), min(h, eY)
                    person_boxes.append((sX, sY, eX, eY))

        unknown_count = 0
        any_unknown = False

        for (sX, sY, eX, eY) in person_boxes:
            face_img = frame[sY:eY, sX:eX]
            if face_img.size == 0:
                continue

            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_face)
            name = "Unknown"
            color = (0, 0, 255)

            for enc in encs:
                if len(known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_face_encodings, enc)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        color = (0, 255, 0)

            cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
            cv2.putText(frame, name, (sX + 5, sY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if name == "Unknown":
                unknown_count += 1
                any_unknown = True

        current_count = unknown_count

        # ---- take snapshot only once every 10 sec ----
        if any_unknown and (time.time() - last_snapshot_time > snapshot_cooldown):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"unknown_{ts}.jpg"
            cv2.imwrite(os.path.join("snapshots", fname), frame)
            last_snapshot_time = time.time()
            print(f"[ALERT] Unknown detected, snapshot saved: {fname}")

        cv2.putText(frame, f"Unknown Count: {unknown_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Security Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    return jsonify({"unknown_count": current_count})


if __name__ == "__main__":
    print("🚀 Running person detection... press 'q' to stop.")
    detect_people()
