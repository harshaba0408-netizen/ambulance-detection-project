import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import pickle
import librosa
import numpy as np
from ultralytics import YOLO
import cv2
import os
from queue import Queue
from PIL import Image, ImageTk
import warnings

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# -------------------------------------------------------------------
# PATHS (make sure the .pt and .pkl files are in the same folder)
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8l_ambulance.pt")
SIREN_CLF_PATH = os.path.join(BASE_DIR, "siren_classifier.pkl")

CONF_THRESHOLD = 0.65
POLICIES = ["AUDIO", "VIDEO", "BOTH"]

# Video display size
VIDEO_W, VIDEO_H = 320, 180


class AmbulanceDetectionApp:
    def __init__(self, root):
        self.root = root
        root.title("AI-Driven Ambulance Traffic Decongestion Dashboard")
        root.geometry("1100x720")
        root.configure(bg="#233047")

        # --- state containers ---
        self.files = {}
        self.audio_detected = {}
        self.video_detected = {}
        self.road_policy = {}

        self.result_queue = Queue()

        self.stop_video_flags = {}
        self.video_threads = {}

        # coordination / latching
        self.latched_green = {}
        self.first_detect_ts = {}
        self.last_light_state = {}

        # UI storage
        self.video_labels = []
        self.policy_vars = []
        self.status_labels = []
        self.signal_canvases = []
        self.signal_circles = []

        # load ML models
        self.load_models()

        # init per-road variables
        for i in range(4):
            road = f"Road {i+1}"
            self.audio_detected[road] = False
            self.video_detected[road] = False
            self.road_policy[road] = "AUDIO"
            self.stop_video_flags[road] = threading.Event()
            self.latched_green[road] = False
            self.first_detect_ts[road] = None
            self.last_light_state[road] = None

        # build UI
        self.build_ui()

        # start queue polling
        self.root.after(100, self.drain_queue)

    # -------------------------------------------------------------------
    def load_models(self):
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model missing: {YOLO_MODEL_PATH}")
        if not os.path.exists(SIREN_CLF_PATH):
            raise FileNotFoundError(f"Siren classifier missing: {SIREN_CLF_PATH}")

        self.yolo_model = YOLO(YOLO_MODEL_PATH)

        with open(SIREN_CLF_PATH, "rb") as f:
            self.audio_clf = pickle.load(f)

        self.class_names = getattr(self.yolo_model, "names", {})

    # -------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------
    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Ambulance Traffic Decongestion Dashboard",
            font=("Helvetica", 22, "bold"),
            bg="#233047",
            fg="white",
        )
        title.pack(pady=10)

        main = tk.Frame(self.root, bg="#233047")
        main.pack(pady=5)

        # 4-road layout
        for i in range(4):
            road = f"Road {i+1}"

            card = tk.LabelFrame(
                main,
                text=road,
                bg="#2c3e50",
                fg="white",
                font=("Helvetica", 13, "bold"),
                padx=8,
                pady=8,
            )
            card.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="n")

            # ---------------------------------------------------
            # FIXED-SIZE VIDEO PLACEHOLDER (CORRECT FIX)
            # ---------------------------------------------------
            video_frame = tk.Frame(card, width=VIDEO_W, height=VIDEO_H, bg="black")
            video_frame.grid_propagate(False)
            video_frame.grid(row=0, column=0, columnspan=3, pady=(0, 6))

            video_label = tk.Label(video_frame, bg="black")
            video_label.place(relx=0.5, rely=0.5, anchor="center")

            self.video_labels.append(video_label)

            # ---------------------------------------------------
            # Buttons
            # ---------------------------------------------------
            ttk.Button(
                card, text="Select Video",
                command=lambda idx=i: self.select_file(idx, "video")
            ).grid(row=1, column=0, padx=5, pady=3)

            ttk.Button(
                card, text="Select Audio",
                command=lambda idx=i: self.select_file(idx, "audio")
            ).grid(row=1, column=1, padx=5, pady=3)

            # Policy dropdown
            pol_var = tk.StringVar(value="AUDIO")
            self.policy_vars.append(pol_var)

            tk.Label(card, text="Policy:", bg="#2c3e50", fg="white").grid(row=2, column=0, sticky="w")

            ttk.OptionMenu(
                card, pol_var, "AUDIO", *POLICIES,
                command=lambda _=None, idx=i: self.on_policy_change(idx)
            ).grid(row=2, column=1, sticky="w")

            # Status text
            status = tk.Label(card, text="Audio: OFF | Video: OFF",
                              bg="#2c3e50", fg="white")
            status.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
            self.status_labels.append(status)

            # Traffic light
            c = tk.Canvas(card, width=60, height=140, bg="black", highlightthickness=0)
            c.grid(row=0, column=3, rowspan=4, padx=(10, 0))

            c.create_rectangle(15, 10, 45, 130, fill="grey25")
            red = c.create_oval(18, 13, 42, 37, fill="red")
            yellow = c.create_oval(18, 53, 42, 77, fill="grey")
            green = c.create_oval(18, 93, 42, 117, fill="grey")

            self.signal_canvases.append(c)
            self.signal_circles.append((red, yellow, green))

        ttk.Button(self.root, text="Clear All", command=self.clear_all).pack(pady=8)

        # log viewer
        log_frame = tk.LabelFrame(self.root, text="Detection Log",
                                  bg="#233047", fg="white")
        log_frame.pack(fill=tk.X, padx=10, pady=5)

        self.log_box = tk.Text(log_frame, height=8, bg="#ecf0f1", state="disabled")
        self.log_box.pack(fill=tk.X)

    # -------------------------------------------------------------------
    # File selection
    # -------------------------------------------------------------------
    def select_file(self, idx, ftype):
        if ftype == "video":
            types = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
        else:
            types = [("Audio files", "*.wav *.mp3")]

        path = filedialog.askopenfilename(filetypes=types)
        if not path:
            return

        road = f"Road {idx+1}"
        self.files[f"road{idx+1}_{ftype}"] = path

        self.log(f"{road}: selected {ftype}: {os.path.basename(path)}")

        policy = self.road_policy[road]

        # apply policy
        if policy == "AUDIO" and ftype == "audio":
            self.start_audio_detection(road, path)

        elif policy == "VIDEO" and ftype == "video":
            self.stop_video(road)
            self.start_video(idx, road, path)

        elif policy == "BOTH":
            if f"road{idx+1}_video" in self.files and f"road{idx+1}_audio" in self.files:
                self.stop_video(road)
                self.start_video(idx, road, self.files[f"road{idx+1}_video"])
                self.start_audio_detection(road, self.files[f"road{idx+1}_audio"])
                self.log(f"{road}: BOTH mode started (video + audio)")

    # -------------------------------------------------------------------
    # Video + YOLO
    # -------------------------------------------------------------------
    def start_video(self, idx, road, path):
        stop_event = threading.Event()
        self.stop_video_flags[road] = stop_event

        def worker():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.log(f"{road}: could not open video")
                return

            frame_no = 0
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # prepare video frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (VIDEO_W, VIDEO_H))
                img = ImageTk.PhotoImage(Image.fromarray(rgb))

                self.root.after(0, lambda i=idx, im=img: self.update_video(i, im))

                # YOLO every 5 frames
                if frame_no % 5 == 0 and not self.video_detected[road]:
                    if self.detect_ambulance(frame):
                        self.result_queue.put(("video", road, True))

                frame_no += 1
                time.sleep(0.01)

            cap.release()

        t = threading.Thread(target=worker, daemon=True)
        self.video_threads[road] = t
        t.start()

    def stop_video(self, road):
        ev = self.stop_video_flags.get(road)
        if ev:
            ev.set()
        t = self.video_threads.get(road)
        if t and t.is_alive():
            t.join(timeout=0.1)

    def update_video(self, idx, img):
        lbl = self.video_labels[idx]
        lbl.imgtk = img
        lbl.configure(image=img)

    def detect_ambulance(self, frame):
        try:
            results = self.yolo_model(frame, verbose=False)
            for r in results:
                for b in getattr(r, "boxes", []):
                    conf = float(b.conf[0])
                    cls = int(b.cls[0])
                    name = str(self.class_names.get(cls, "")).lower()
                    if conf >= CONF_THRESHOLD and "ambulance" in name:
                        return True
        except Exception as e:
            self.log(f"YOLO error: {e}")
        return False

    # -------------------------------------------------------------------
    # Audio
    # -------------------------------------------------------------------
    def start_audio_detection(self, road, path):
        def worker():
            try:
                y, sr = librosa.load(path, duration=3.0)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                feat = np.mean(mfccs.T, axis=0)
                pred = int(self.audio_clf.predict([feat])[0])
                siren = pred == 1
            except Exception as e:
                self.log(f"{road}: audio error: {e}")
                siren = False

            if siren and not self.audio_detected[road]:
                self.result_queue.put(("audio", road, True))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------------
    # Queue handler
    # -------------------------------------------------------------------
    def drain_queue(self):
        changed = False

        while not self.result_queue.empty():
            kind, road, val = self.result_queue.get()
            if kind == "audio" and not self.audio_detected[road]:
                self.audio_detected[road] = True
                changed = True
                self.log(f"{road}: SIREN detected")

            if kind == "video" and not self.video_detected[road]:
                self.video_detected[road] = True
                changed = True
                self.log(f"{road}: AMBULANCE detected")

        if changed:
            self.recompute_signals()

        self.root.after(100, self.drain_queue)

    # -------------------------------------------------------------------
    # Policy change
    # -------------------------------------------------------------------
    def on_policy_change(self, idx):
        road = f"Road {idx+1}"
        self.road_policy[road] = self.policy_vars[idx].get()
        self.recompute_signals()

    # -------------------------------------------------------------------
    # Traffic light control logic
    # -------------------------------------------------------------------
    def recompute_signals(self):
        candidates = []

        for i in range(4):
            road = f"Road {i+1}"
            policy = self.road_policy[road]
            a = self.audio_detected[road]
            v = self.video_detected[road]

            if (policy == "AUDIO" and a) or (policy == "VIDEO" and v) or (policy == "BOTH" and a and v):
                if not self.latched_green[road]:
                    self.latched_green[road] = True
                    self.first_detect_ts[road] = time.time()

                candidates.append(road)

        if not candidates:
            for i in range(4):
                r = f"Road {i+1}"
                self.set_light(r, False)
                self.update_status(r)
            return

        # choose earliest triggered road
        chosen = min(candidates, key=lambda r: self.first_detect_ts[r])

        for i in range(4):
            r = f"Road {i+1}"
            self.set_light(r, r == chosen)
            self.update_status(r)

    # -------------------------------------------------------------------
    # UI lighting
    # -------------------------------------------------------------------
    def set_light(self, road, green_on):
        idx = int(road.split()[-1]) - 1
        canvas = self.signal_canvases[idx]
        red, yellow, green = self.signal_circles[idx]

        prev = self.last_light_state.get(road)
        if prev == green_on:
            return

        if green_on:
            canvas.itemconfig(red, fill="grey")
            canvas.itemconfig(green, fill="green")
        else:
            canvas.itemconfig(red, fill="red")
            canvas.itemconfig(green, fill="grey")

        self.last_light_state[road] = green_on

    def update_status(self, road):
        idx = int(road.split()[-1]) - 1
        a = "ON" if self.audio_detected[road] else "OFF"
        v = "ON" if self.video_detected[road] else "OFF"
        self.status_labels[idx].config(text=f"Audio: {a} | Video: {v}")

    # -------------------------------------------------------------------
    def clear_all(self):
        for i in range(4):
            road = f"Road {i+1}"

            self.stop_video(road)
            self.audio_detected[road] = False
            self.video_detected[road] = False
            self.latched_green[road] = False
            self.first_detect_ts[road] = None
            self.last_light_state[road] = None

            self.road_policy[road] = "AUDIO"
            self.policy_vars[i].set("AUDIO")

            self.files.pop(f"road{i+1}_video", None)
            self.files.pop(f"road{i+1}_audio", None)

            self.set_light(road, False)
            self.update_status(road)

        self.log("All signals cleared")

    # -------------------------------------------------------------------
    def log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("1.0", f"{t} - {msg}\n")
        self.log_box.config(state="disabled")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AmbulanceDetectionApp(root)
    root.mainloop()