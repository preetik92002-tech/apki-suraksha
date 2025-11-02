 # security_server.py
import threading, time, os
from flask import Flask, request, jsonify, Response, send_from_directory
import cv2, numpy as np, face_recognition
from datetime import datetime
from supabase import create_client, Client

# ---------------- SUPABASE CONFIG ----------------
SUPABASE_URL = "https://rvitfkwqmtogfdvdsuhc.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ2aXRma3dxbXRvZ2ZkdmRzdWhjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwMTQ5NjEsImV4cCI6MjA3NTU5MDk2MX0.XE7L3oxFtjWpv7Y8RzMYmuSlyBI-Yl-PMHNrNLSWO5k"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- FLASK APP SETUP ----------------
app = Flask(__name__)

MODEL_PROTOTXT = "deploy.prototxt"
MODEL_WEIGHTS = "mobilenet_iter_73000.caffemodel"
KNOWN_DIR = "known_faces"
SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

detection_thread = None
detection_thread_stop = threading.Event()
detection_running = False
current_count = 0
threshold = 3
last_snapshot = None

# Load MobileNet SSD
net = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_WEIGHTS)

# Load known faces
known_face_encodings = []
known_face_names = []
if os.path.isdir(KNOWN_DIR):
    for f in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, f)
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if len(enc) > 0:
                known_face_encodings.append(enc[0])
                known_face_names.append(os.path.splitext(f)[0])
print("✅ Loaded known faces:", known_face_names)

# ---------------- DETECTION LOOP ----------------
def detection_loop():
    global current_count, last_snapshot, detection_running
    cap = cv2.VideoCapture(0)
    detection_running = True
    print("[DETECT] Detection started...")

    last_alert_time = 0
    alert_cooldown = 10  # seconds between snapshots

    while not detection_thread_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        persons = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > 0.5:
                cls = int(detections[0, 0, i, 1])
                if cls == 15:
                    box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                    sX, sY, eX, eY = box
                    sX, sY, eX, eY = max(0, sX), max(0, sY), min(w, eX), min(h, eY)
                    persons.append((sX, sY, eX, eY))

        unknown_count = 0
        for (sX, sY, eX, eY) in persons:
            face_img = frame[sY:eY, sX:eX]
            if face_img.size == 0:
                continue
            rgb_small = cv2.cvtColor(cv2.resize(face_img, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_small)
            matched = False
            name = "Unknown"
            for enc in encs:
                matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.45)
                if True in matches:
                    matched = True
                    name = known_face_names[matches.index(True)]
                    break

            color = (0, 255, 0) if matched else (0, 0, 255)
            cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
            cv2.putText(frame, name, (sX, sY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if not matched:
                unknown_count += 1

        current_count = unknown_count

        # --- Only snapshot if unknown and cooldown passed ---
        now = time.time()
        if unknown_count > 0 and (now - last_alert_time > alert_cooldown):
            last_alert_time = now
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"alert_{ts}.jpg"
            fpath = os.path.join(SNAP_DIR, fname)
            cv2.imwrite(fpath, frame)
            last_snapshot = fname
            print(f"[ALERT] Unknown detected, snapshot saved {fname}")
            # Upload to Supabase alerts
            try:
                supabase.table("alerts").insert({
                    "message": f"⚠️ Unknown detected at {ts}",
                    "image_url": f"http://127.0.0.1:5000/snapshots/{fname}"
                }).execute()
                print("✅ Alert saved to Supabase")
            except Exception as e:
                print("❌ Supabase error:", e)

        cv2.imshow("Security Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detection_thread_stop.set()
            break
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    detection_running = False
    print("[DETECT] Detection stopped.")

# ---------------- API ROUTES ----------------
@app.route("/start_detection", methods=["POST"])
def start_detection():
    global detection_thread
    if detection_running:
        return jsonify({"message": "already_running"})
    detection_thread_stop.clear()
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    return jsonify({"message": "detection_started"})

@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    detection_thread_stop.set()
    return jsonify({"message": "detection_stopped"})

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "running": detection_running,
        "current_count": current_count,
        "threshold": threshold,
        "last_snapshot": last_snapshot
    })

@app.route("/snapshots/<path:fname>")
def get_snapshot(fname):
    return send_from_directory(SNAP_DIR, fname)

@app.route("/video_feed")
def video_feed():
    def gen_frames():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.putText(frame, f"Unknown: {current_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
        cap.release()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
