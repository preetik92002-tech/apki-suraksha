from flask import Flask, jsonify
import cv2
import numpy as np
import threading, time, os
from datetime import datetime
from supabase import create_client, Client

# --- Supabase Setup ---
SUPABASE_URL = "https://rvitfkwqmtogfdvdsuhc.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ2aXRma3dxbXRvZ2ZkdmRzdWhjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwMTQ5NjEsImV4cCI6MjA3NTU5MDk2MX0.XE7L3oxFtjWpv7Y8RzMYmuSlyBI-Yl-PMHNrNLSWO5k"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Flask App ---
app = Flask(__name__)

# --- Model and Config ---
MODEL_PROTOTXT = "deploy.prototxt"
MODEL_WEIGHTS = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_WEIGHTS)
threshold = 3
current_count = 0
video = cv2.VideoCapture(0)
alert_triggered = False

# --- Directory for snapshots ---
os.makedirs("snapshots", exist_ok=True)

def detect_people():
    global current_count, alert_triggered
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Person detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:
                    count += 1
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        current_count = count
        cv2.putText(frame, f"People Count: {current_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # --- Alert & Snapshot ---
        if current_count > threshold and not alert_triggered:
            alert_triggered = True
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = f"snapshots/alert_{ts}.jpg"
            cv2.imwrite(snapshot_path, frame)

            data = {
                "message": f"⚠️ Alert: {current_count} people detected (threshold {threshold})",
                "image_url": f"http://127.0.0.1:5000/{snapshot_path}"
            }
            try:
                supabase.table("alerts").insert(data).execute()
                print("✅ Alert logged in Supabase!")
            except Exception as e:
                print("❌ Error uploading alert:", e)

        elif current_count <= threshold:
            alert_triggered = False

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

@app.route('/status')
def status():
    return jsonify({"people_count": current_count, "threshold": threshold})

if __name__ == "__main__":
    t = threading.Thread(target=detect_people)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000)
