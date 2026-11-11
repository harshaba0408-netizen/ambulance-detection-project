"""Event schema shared by detection, audio, fusion and signal components.

Every decision in the system is traceable to one or more `DetectionEvent`
objects. The schema is intentionally flat and JSON-serialisable so it can be
written to a log, replayed in tests, or inspected by the dashboard without
importing Torch, Librosa or Tkinter.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class EventSource(str, Enum):
    """Sensor modality that produced the evidence."""

    VIDEO = "video"
    AUDIO = "audio"


class DetectionType(str, Enum):
    """What was detected. Kept explicit so future classes do not collide."""

    AMBULANCE = "ambulance"
    SIREN = "siren"


class EventStatus(str, Enum):
    """Lifecycle of a single piece of evidence."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    RELEASED = "released"
    EXPIRED = "expired"


def new_event_id() -> str:
    """Return a process-unique event identifier."""
    return uuid.uuid4().hex


def utc_iso(timestamp: float) -> str:
    """Render an epoch timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _validate_timestamp(timestamp: float) -> None:
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ValueError("timestamp must be a number")
    if not math.isfinite(float(timestamp)) or float(timestamp) < 0.0:
        raise ValueError(f"timestamp must be finite and >= 0, got {timestamp!r}")


def _validate_confidence(confidence: float) -> None:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence must be within [0.0, 1.0], got {confidence!r}")


@dataclass(frozen=True)
class DetectionEvent:
    """One timestamped piece of evidence for one approach.

    Video events carry an optional bounding box and frame index. Audio events
    carry the window start/end in seconds relative to the stream. Both fields
    are optional because the schema is shared, but validation checks whichever
    fields are populated.
    """

    approach: str
    source: EventSource
    detection_type: DetectionType
    confidence: float
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=new_event_id)
    status: EventStatus = EventStatus.CANDIDATE
    evidence_refs: Tuple[str, ...] = ()
    bbox: Optional[Tuple[float, float, float, float]] = None
    frame_index: Optional[int] = None
    window_start_s: Optional[float] = None
    window_end_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.approach, str) or not self.approach.strip():
            raise ValueError("approach must be a non-empty string")
        if not isinstance(self.source, EventSource):
            raise ValueError("source must be an EventSource")
        if not isinstance(self.detection_type, DetectionType):
            raise ValueError("detection_type must be a DetectionType")
        if not isinstance(self.status, EventStatus):
            raise ValueError("status must be an EventStatus")
        _validate_confidence(self.confidence)
        _validate_timestamp(self.timestamp)
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(self.evidence_refs, tuple) or any(
            not isinstance(ref, str) or not ref.strip() for ref in self.evidence_refs
        ):
            raise ValueError("evidence_refs must be a tuple of non-empty strings")
        if self.frame_index is not None and (isinstance(self.frame_index, bool) or self.frame_index < 0):
            raise ValueError("frame_index must be >= 0 when provided")
        if self.bbox is not None:
            if len(self.bbox) != 4 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in self.bbox
            ):
                raise ValueError("bbox must be four finite numbers (x1, y1, x2, y2)")
            x1, y1, x2, y2 = self.bbox
            if x2 < x1 or y2 < y1:
                raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1")
        if (self.window_start_s is None) != (self.window_end_s is None):
            raise ValueError("window_start_s and window_end_s must be provided together")
        if self.window_start_s is not None and self.window_end_s is not None:
            for name, value in (
                ("window_start_s", self.window_start_s),
                ("window_end_s", self.window_end_s),
            ):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
                if value < 0.0:
                    raise ValueError(f"{name} must be >= 0")
            if self.window_end_s <= self.window_start_s:
                raise ValueError("window_end_s must be greater than window_start_s")

    @classmethod
    def video(
        cls,
        approach: str,
        confidence: float,
        *,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
        status: EventStatus = EventStatus.CANDIDATE,
        event_id: Optional[str] = None,
        evidence_refs: Tuple[str, ...] = (),
    ) -> "DetectionEvent":
        return cls(
            approach=approach,
            source=EventSource.VIDEO,
            detection_type=DetectionType.AMBULANCE,
            confidence=confidence,
            timestamp=time.time() if timestamp is None else timestamp,
            event_id=new_event_id() if event_id is None else event_id,
            status=status,
            evidence_refs=evidence_refs,
            bbox=bbox,
            frame_index=frame_index,
        )

    @classmethod
    def audio(
        cls,
        approach: str,
        confidence: float,
        *,
        window_start_s: Optional[float] = None,
        window_end_s: Optional[float] = None,
        timestamp: Optional[float] = None,
        status: EventStatus = EventStatus.CANDIDATE,
        event_id: Optional[str] = None,
        evidence_refs: Tuple[str, ...] = (),
    ) -> "DetectionEvent":
        return cls(
            approach=approach,
            source=EventSource.AUDIO,
            detection_type=DetectionType.SIREN,
            confidence=confidence,
            timestamp=time.time() if timestamp is None else timestamp,
            event_id=new_event_id() if event_id is None else event_id,
            status=status,
            evidence_refs=evidence_refs,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
        )

    def with_status(self, status: EventStatus) -> "DetectionEvent":
        """Return a copy with a new lifecycle status."""
        return replace(self, status=status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": float(self.timestamp),
            "timestamp_iso": utc_iso(float(self.timestamp)),
            "approach": self.approach,
            "source": self.source.value,
            "detection_type": self.detection_type.value,
            "confidence": float(self.confidence),
            "status": self.status.value,
            "evidence_refs": list(self.evidence_refs),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "frame_index": self.frame_index,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectionEvent":
        if not isinstance(data, Mapping):
            raise ValueError("event data must be a mapping")
        bbox = data.get("bbox")
        return cls(
            approach=data["approach"],
            source=EventSource(data["source"]),
            detection_type=DetectionType(data["detection_type"]),
            confidence=float(data["confidence"]),
            timestamp=float(data.get("timestamp", time.time())),
            event_id=data.get("event_id") or new_event_id(),
            status=EventStatus(data.get("status", EventStatus.CANDIDATE.value)),
            evidence_refs=tuple(data.get("evidence_refs", ())),
            bbox=tuple(bbox) if bbox is not None else None,
            frame_index=data.get("frame_index"),
            window_start_s=data.get("window_start_s"),
            window_end_s=data.get("window_end_s"),
        )


@dataclass(frozen=True)
class PriorityRequest:
    """Result of fusion for one approach: should priority be requested now?"""

    approach: str
    active: bool
    reason: str
    timestamp: float = field(default_factory=time.time)
    event_ids: Tuple[str, ...] = ()
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.approach, str) or not self.approach.strip():
            raise ValueError("approach must be a non-empty string")
        if not isinstance(self.active, bool):
            raise ValueError("active must be a bool")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        _validate_timestamp(self.timestamp)
        if not isinstance(self.event_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.event_ids
        ):
            raise ValueError("event_ids must be a tuple of non-empty strings")
        if self.confidence is not None:
            _validate_confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": self.approach,
            "active": self.active,
            "reason": self.reason,
            "timestamp": float(self.timestamp),
            "timestamp_iso": utc_iso(float(self.timestamp)),
            "event_ids": list(self.event_ids),
            "confidence": float(self.confidence) if self.confidence is not None else None,
        }
