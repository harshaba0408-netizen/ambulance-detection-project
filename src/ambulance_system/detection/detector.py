"""Thin, testable wrapper around the Ultralytics YOLO detector.

The wrapper does three things and nothing else:

1. load one checkpoint,
2. run prediction with explicit thresholds,
3. convert raw boxes into validated `Detection` objects.

Model loading is injectable (`model=`) so the parsing and filtering logic can
be tested without loading an 87 MB checkpoint.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ambulance_system.detection.types import Detection
from ambulance_system.logging_utils import get_logger


def _first_number(value: Any) -> float:
    array = np.asarray(value.cpu() if hasattr(value, "cpu") else value).reshape(-1)
    if array.size == 0:
        raise ValueError("empty box array")
    return float(array[0])


def _to_floats(value: Any) -> list[float]:
    array = np.asarray(value.cpu() if hasattr(value, "cpu") else value).reshape(-1)
    return [float(item) for item in array]


class Detector:
    """YOLO inference wrapper with explicit thresholds and class filtering."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model: Any = None,
        confidence_threshold: float = 0.65,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        class_filter: str = "ambulance",
    ) -> None:
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(float(confidence_threshold))
            or not 0.0 <= float(confidence_threshold) <= 1.0
        ):
            raise ValueError("confidence_threshold must be a finite number within [0.0, 1.0]")
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, (int, float))
            or not math.isfinite(float(iou_threshold))
            or not 0.0 <= float(iou_threshold) <= 1.0
        ):
            raise ValueError("iou_threshold must be a finite number within [0.0, 1.0]")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("imgsz must be a positive int")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(class_filter, str) or not class_filter.strip():
            raise ValueError("class_filter must be a non-empty string")

        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._imgsz = imgsz
        self._device = device
        self._class_filter = class_filter.strip().lower()
        self._logger = get_logger("detection")

        if model is None:
            path = Path(model_path)
            if not path.is_file():
                raise FileNotFoundError(f"YOLO model missing: {path}")
            from ultralytics import YOLO

            model = YOLO(str(path))
        self._model = model
        self._names = dict(getattr(model, "names", {}) or {})

    # ------------------------------------------------------------------
    @property
    def model(self) -> Any:
        return self._model

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    # ------------------------------------------------------------------
    def detect(
        self,
        frame: Any,
        *,
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> list[Detection]:
        """Run inference on one BGR/RGB frame and return filtered detections."""
        self._validate_frame(frame)
        results = self._model.predict(
            source=frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in self._as_result_list(results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                detection = self._parse_box(box, frame_index, timestamp)
                if detection is not None:
                    detections.append(detection)

        detections.sort(key=lambda item: (-item.confidence, item.bbox))
        return detections

    def detect_best(
        self,
        frame: Any,
        *,
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[Detection]:
        """Return the highest-confidence filtered detection, if any."""
        detections = self.detect(frame, frame_index=frame_index, timestamp=timestamp)
        return detections[0] if detections else None

    def __call__(self, frame: Any, **kwargs: Any) -> list[Detection]:
        return self.detect(frame, **kwargs)

    # ------------------------------------------------------------------
    def _parse_box(
        self,
        box: Any,
        frame_index: Optional[int],
        timestamp: Optional[float],
    ) -> Optional[Detection]:
        try:
            xyxy = _to_floats(box.xyxy[0])
            confidence = _first_number(box.conf)
            class_id = int(_first_number(box.cls))
        except (AttributeError, IndexError, TypeError, ValueError):
            self._logger.warning("skipping malformed detection box")
            return None

        class_name = str(self._names.get(class_id, class_id))
        if self._class_filter not in class_name.lower():
            return None
        if confidence < self._confidence_threshold:
            return None
        if len(xyxy) != 4:
            self._logger.warning("skipping detection with %d bbox values", len(xyxy))
            return None

        try:
            return Detection(
                bbox=tuple(xyxy),  # type: ignore[arg-type]
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                frame_index=frame_index,
                timestamp=timestamp if timestamp is not None else time.time(),
            )
        except ValueError as exc:
            self._logger.warning("skipping invalid detection: %s", exc)
            return None

    @staticmethod
    def _as_result_list(results: Any) -> list[Any]:
        if results is None:
            return []
        if isinstance(results, (list, tuple)):
            return list(results)
        return [results]

    @staticmethod
    def _validate_frame(frame: Any) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array")
        if frame.ndim != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty H x W x C array")
        if frame.shape[2] not in (1, 3, 4):
            raise ValueError("frame must have 1, 3 or 4 channels")
