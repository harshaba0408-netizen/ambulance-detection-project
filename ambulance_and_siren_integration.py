import pickle
import librosa
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import time
import os
import sys

# -----------------------
# BASE DIRECTORY (IMPORTANT FIX)
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_VIDEO = os.path.join(BASE_DIR, "data", "Video", "1.mp4")
# Optional CLI override: python ambulance_and_siren_integration.py <video_path>
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolov8l_ambulance.pt")
SIREN_MODEL_PATH = os.path.join(BASE_DIR, "siren_classifier.pkl")
SIREN_AUDIO_PATH = os.path.join(BASE_DIR, "data/audio/siren/sound_1.wav")

LOG_PATH = os.path.join(BASE_DIR, "emergency_detection_log.txt")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

os.makedirs(EVIDENCE_DIR, exist_ok=True)

# -----------------------
# SETTINGS
# -----------------------
AUDIO_CHECK_INTERVAL = 5.0
VIS_CONF_THRESH = 0.70
VIS_HISTORY_LEN = 5
VIS_REQUIRED = 2
AUDIO_HISTORY_LEN = 5
AUDIO_REQUIRED = 1

# -----------------------
# CHECK FILES EXIST
# -----------------------
def check_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} NOT FOUND at: {path}")

check_file(YOLO_WEIGHTS, "YOLO model")
check_file(SIREN_MODEL_PATH, "Siren classifier")
check_file(VIDEO_PATH, "Video file")

# -----------------------
# LOAD MODELS
# -----------------------
print("Loading siren classifier...")
with open(SIREN_MODEL_PATH, 'rb') as f:
    audio_clf = pickle.load(f)

print("Loading YOLO model...")
model = YOLO(YOLO_WEIGHTS)

# -----------------------
# FUNCTIONS
# -----------------------
def extract_mfcc_from_file(audio_path, duration=3):
    y, sr = librosa.load(audio_path, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def save_evidence_frame(img):
    ts = int(time.time())
    path = os.path.join(EVIDENCE_DIR, f"evidence_{ts}.jpg")
    cv2.imwrite(path, img)
    return path

# -----------------------
# STATE VARIABLES
# -----------------------
last_audio_check = 0.0
audio_history = []
visual_history = []

log_file = open(LOG_PATH, 'a')

# -----------------------
# VIDEO CAPTURE
# -----------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

# -----------------------
# MAIN LOOP
# -----------------------
try:
    while True:
        success, img = cap.read()
        if not success:
            print("End of video")
            break

        # -----------------------
        # VISION DETECTION
        # -----------------------
        ambulance_detected = False

        results = model(img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = r.names[cls].lower()

                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
                cvzone.putTextRect(img, f"{label} {conf:.2f}", (x1, y1))

                if "ambulance" in label and conf >= VIS_CONF_THRESH:
                    ambulance_detected = True

        # Visual history
        visual_history.append(1 if ambulance_detected else 0)
        if len(visual_history) > VIS_HISTORY_LEN:
            visual_history.pop(0)

        # -----------------------
        # AUDIO DETECTION
        # -----------------------
        siren_detected = False
        now = time.time()

        if now - last_audio_check >= AUDIO_CHECK_INTERVAL:
            try:
                features = extract_mfcc_from_file(SIREN_AUDIO_PATH)
                pred = audio_clf.predict([features])[0]
                siren_detected = (pred == 1)
            except Exception as e:
                print("Audio error:", e)

            audio_history.append(1 if siren_detected else 0)
            if len(audio_history) > AUDIO_HISTORY_LEN:
                audio_history.pop(0)

            last_audio_check = now
        else:
            siren_detected = sum(audio_history) >= AUDIO_REQUIRED

        # -----------------------
        # DECISION LOGIC
        # -----------------------
        visual_ok = sum(visual_history) >= VIS_REQUIRED
        audio_ok = sum(audio_history) >= AUDIO_REQUIRED

        if visual_ok and audio_ok:
            status = "🚨 CONFIRMED EMERGENCY"
            save_evidence_frame(img)
        elif visual_ok:
            status = "Ambulance detected"
        elif audio_ok:
            status = "Siren detected"
        else:
            status = "No detection"

        # -----------------------
        # DISPLAY
        # -----------------------
        color = (0, 255, 0) if "CONFIRMED" in status else (0, 0, 255)
        cv2.putText(img, status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Detection", img)

        log_file.write(f"{time.time()} - {status}\n")
        log_file.flush()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print("Finished cleanly")