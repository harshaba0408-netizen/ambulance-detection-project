from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ambulance_system.detection import Detection, Detector

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "yolov8l_ambulance.pt"


class FakeBox:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.asarray([xyxy], dtype=float)
        self.conf = np.asarray([confidence], dtype=float)
        self.cls = np.asarray([class_id], dtype=float)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    names = {0: "ambulance", 1: "car"}

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.boxes)]


def frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def detector(boxes, **kwargs) -> Detector:
    return Detector(
        "unused.pt",
        model=FakeModel(boxes),
        confidence_threshold=kwargs.pop("confidence_threshold", 0.65),
        imgsz=kwargs.pop("imgsz", 640),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------
def test_detection_serialises_and_converts_to_event():
    detection = Detection(
        bbox=(10.0, 20.0, 110.0, 120.0),
        confidence=0.88,
        class_id=0,
        class_name="ambulance",
        frame_index=7,
        timestamp=123.0,
    )

    payload = detection.to_dict()
    event = detection.to_event("Road 1")

    assert payload["bbox"] == [10.0, 20.0, 110.0, 120.0]
    assert event.approach == "Road 1"
    assert event.source.value == "video"
    assert event.bbox == (10.0, 20.0, 110.0, 120.0)
    assert event.frame_index == 7
    assert event.timestamp == pytest.approx(123.0)


def test_detection_rejects_inverted_bbox():
    with pytest.raises(ValueError, match="bbox"):
        Detection((100.0, 0.0, 10.0, 50.0), 0.9, 0, "ambulance")


def test_detection_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        Detection((0.0, 0.0, 10.0, 10.0), 1.5, 0, "ambulance")


def test_detection_rejects_negative_frame_index():
    with pytest.raises(ValueError, match="frame_index"):
        Detection((0.0, 0.0, 10.0, 10.0), 0.9, 0, "ambulance", frame_index=-1)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="YOLO model missing"):
        Detector(tmp_path / "missing.pt")


def test_invalid_constructor_values():
    with pytest.raises(ValueError, match="confidence_threshold"):
        Detector("unused.pt", model=FakeModel([]), confidence_threshold=2.0)
    with pytest.raises(ValueError, match="imgsz"):
        Detector("unused.pt", model=FakeModel([]), imgsz=0)
    with pytest.raises(ValueError, match="class_filter"):
        Detector("unused.pt", model=FakeModel([]), class_filter="  ")


def test_published_checkpoint_loads():
    wrapped = Detector(MODEL_PATH)

    assert wrapped.model is not None
    assert 0 in wrapped.class_names
    assert "ambulance" in wrapped.class_names[0].lower()


# ---------------------------------------------------------------------------
# Filtering and parsing
# ---------------------------------------------------------------------------
def test_filters_low_confidence_and_other_classes():
    wrapped = detector(
        [
            FakeBox((0, 0, 10, 10), 0.90, 0),
            FakeBox((10, 10, 20, 20), 0.30, 0),
            FakeBox((20, 20, 30, 30), 0.99, 1),
        ]
    )

    detections = wrapped.detect(frame(), timestamp=1.0)

    assert len(detections) == 1
    assert detections[0].class_name == "ambulance"
    assert detections[0].confidence == pytest.approx(0.90)
    assert detections[0].bbox == (0.0, 0.0, 10.0, 10.0)


def test_results_are_sorted_by_confidence():
    wrapped = detector(
        [
            FakeBox((0, 0, 10, 10), 0.70, 0),
            FakeBox((20, 20, 30, 30), 0.95, 0),
            FakeBox((40, 40, 50, 50), 0.80, 0),
        ]
    )

    detections = wrapped.detect(frame())

    assert [round(item.confidence, 2) for item in detections] == [0.95, 0.80, 0.70]


def test_detect_best_returns_none_without_detections():
    wrapped = detector([])

    assert wrapped.detect(frame()) == []
    assert wrapped.detect_best(frame()) is None


def test_detect_best_returns_highest_confidence():
    wrapped = detector(
        [
            FakeBox((0, 0, 10, 10), 0.70, 0),
            FakeBox((20, 20, 30, 30), 0.95, 0),
        ]
    )

    best = wrapped.detect_best(frame(), frame_index=3)

    assert best is not None
    assert best.confidence == pytest.approx(0.95)
    assert best.frame_index == 3


def test_malformed_box_is_skipped():
    class BrokenBox:
        pass

    wrapped = detector([BrokenBox(), FakeBox((0, 0, 10, 10), 0.9, 0)])

    detections = wrapped.detect(frame())

    assert len(detections) == 1


def test_inverted_model_box_is_skipped():
    wrapped = detector([FakeBox((100, 0, 10, 50), 0.9, 0)])

    assert wrapped.detect(frame()) == []


def test_prediction_kwargs_are_explicit():
    fake = FakeModel([])
    wrapped = Detector(
        "unused.pt",
        model=fake,
        confidence_threshold=0.7,
        iou_threshold=0.5,
        imgsz=320,
        device="cpu",
    )

    wrapped.detect(frame())

    call = fake.calls[0]
    assert call["conf"] == pytest.approx(0.7)
    assert call["iou"] == pytest.approx(0.5)
    assert call["imgsz"] == 320
    assert call["device"] == "cpu"
    assert call["verbose"] is False


def test_frames_must_be_non_empty_three_channel_arrays():
    wrapped = detector([])

    with pytest.raises(TypeError, match="numpy array"):
        wrapped.detect([[0, 0, 0]])
    with pytest.raises(ValueError, match="non-empty"):
        wrapped.detect(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="channels"):
        wrapped.detect(np.zeros((10, 10, 2), dtype=np.uint8))


def test_call_is_an_alias_for_detect():
    wrapped = detector([FakeBox((0, 0, 10, 10), 0.9, 0)])

    assert len(wrapped(frame())) == 1
