import os
import math
import cv2
import cvzone
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8l_ambulance.pt")

# Fallback video selection: local sample or project video
DEFAULT_VIDEO = os.path.join(BASE_DIR, "data", "Video", "1.mp4")
LOCAL_PREVIEW = os.path.join(BASE_DIR, "3599822461-preview.mp4")
VIDEO_PATH = LOCAL_PREVIEW if os.path.exists(LOCAL_PREVIEW) else DEFAULT_VIDEO


def run_ambulance_test(video_source=VIDEO_PATH):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    model = YOLO(MODEL_PATH)
    class_names = ['ambulance']

    print(f"Running ambulance visual detection on {video_source}... Press 'q' to quit.")
    while True:
        success, img = cap.read()
        if not success:
            print("End of video stream.")
            break

        results = model(img, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls_id = int(box.cls[0])
                label = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
                cvzone.putTextRect(img, f'{label} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv2.imshow("Ambulance Detection Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_ambulance_test()
