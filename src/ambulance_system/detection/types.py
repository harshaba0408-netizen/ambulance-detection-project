"""Detection result types shared by the vision pipeline and evaluation tools."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from ambulance_system.events import DetectionEvent


@dataclass(frozen=True)
class Detection:
    """One filtered detection in pixel coordinates.

    Coordinates are absolute pixels in the source frame, ordered
    `(x1, y1, x2, y2)`. They are not normalised; conversion is the caller's
    responsibility.
    """

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    frame_index: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, tuple) or len(self.bbox) != 4:
            raise ValueError("bbox must be a tuple of four numbers (x1, y1, x2, y2)")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.bbox
        ):
            raise ValueError("bbox values must be finite numbers")
        x1, y1, x2, y2 = (float(value) for value in self.bbox)
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be a finite number within [0.0, 1.0]")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative int")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string")
        if self.frame_index is not None and (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative int when provided")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
            or float(self.timestamp) < 0.0
        ):
            raise ValueError("timestamp must be finite and >= 0")

    def to_event(self, approach: str) -> DetectionEvent:
        """Convert to the shared event schema used by fusion and logging."""
        return DetectionEvent.video(
            approach=approach,
            confidence=float(self.confidence),
            bbox=self.bbox,
            frame_index=self.frame_index,
            timestamp=float(self.timestamp),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [float(value) for value in self.bbox],
            "confidence": float(self.confidence),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "frame_index": self.frame_index,
            "timestamp": float(self.timestamp),
        }
