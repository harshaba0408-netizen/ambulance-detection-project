import os
import pytest
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8l_ambulance.pt")
TEST_IMAGE_PATH = os.path.join(
    BASE_DIR, "data", "ambulance_images", "test", "images", "A39_jpg.rf.93ba33744d81d5e413f43c2358d97d3c.jpg"
)


def test_yolo_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model weight file missing: {MODEL_PATH}"


def test_yolo_model_loading():
    model = YOLO(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "names")
    assert 0 in model.names
    assert "ambulance" in model.names[0].lower()


def test_yolo_inference_on_synthetic_frame():
    model = YOLO(MODEL_PATH)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(dummy_frame, verbose=False)
    assert len(results) == 1
    assert hasattr(results[0], "boxes")


def test_yolo_inference_on_real_ambulance_image():
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image not available: {TEST_IMAGE_PATH}")

    model = YOLO(MODEL_PATH)
    results = model(TEST_IMAGE_PATH, verbose=False)
    assert len(results) == 1
    boxes = results[0].boxes
    assert len(boxes) > 0, "Expected at least 1 ambulance detection in test image"
    conf = float(boxes.conf[0])
    cls_id = int(boxes.cls[0])
    assert cls_id == 0
    assert conf >= 0.70, f"Detection confidence {conf:.2f} below 0.70 threshold"


def test_yolo_invalid_model_path():
    with pytest.raises(Exception):
        _ = YOLO("non_existent_weights_12345.pt")
