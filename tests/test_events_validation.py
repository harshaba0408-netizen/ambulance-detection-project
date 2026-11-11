"""Validation tests for the shared event schema.

The happy paths live in `test_events.py`; these cases exercise the rejection
branches so malformed evidence cannot reach fusion or the signal controller.
"""

from __future__ import annotations

import pytest

from ambulance_system.events import (
    DetectionEvent,
    DetectionType,
    EventSource,
    EventStatus,
    PriorityRequest,
    utc_iso,
)


def event_kwargs(**overrides):
    base = dict(
        approach="Road 1",
        source=EventSource.VIDEO,
        detection_type=DetectionType.AMBULANCE,
        confidence=0.5,
        timestamp=1.0,
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"source": "video"}, "source"),
        ({"detection_type": "ambulance"}, "detection_type"),
        ({"status": "candidate"}, "status"),
        ({"confidence": "0.5"}, "confidence must be a number"),
        ({"timestamp": "now"}, "timestamp must be a number"),
        ({"timestamp": True}, "timestamp must be a number"),
        ({"event_id": "   "}, "event_id"),
        ({"evidence_refs": ["evt-1"]}, "evidence_refs"),
        ({"evidence_refs": ("evt-1", " ")}, "evidence_refs"),
        ({"frame_index": True}, "frame_index"),
        ({"frame_index": -1}, "frame_index"),
        ({"bbox": (1.0, 2.0, 3.0)}, "bbox"),
        ({"bbox": (1.0, 2.0, 3.0, "4")}, "bbox"),
        ({"bbox": (1.0, 2.0, float("nan"), 4.0)}, "bbox"),
        ({"bbox": (0.0, 0.0, True, 1.0)}, "bbox"),
        ({"window_start_s": 1.0}, "provided together"),
        ({"window_start_s": -1.0, "window_end_s": 1.0}, "window_start_s must be >= 0"),
        ({"window_start_s": True, "window_end_s": 1.0}, "window_start_s must be a finite number"),
        ({"window_start_s": 0.0, "window_end_s": float("inf")}, "window_end_s must be a finite number"),
        ({"window_start_s": 2.0, "window_end_s": 2.0}, "greater than"),
    ],
)
def test_detection_event_rejects_malformed_fields(overrides, match):
    with pytest.raises(ValueError, match=match):
        DetectionEvent(**event_kwargs(**overrides))


def test_from_dict_rejects_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        DetectionEvent.from_dict([("approach", "Road 1")])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"approach": "  "}, "approach"),
        ({"timestamp": -1.0}, "timestamp"),
        ({"timestamp": True}, "timestamp must be a number"),
        ({"event_ids": ["evt-1"]}, "event_ids"),
        ({"event_ids": ("",)}, "event_ids"),
        ({"confidence": 1.5}, "confidence"),
        ({"confidence": True}, "confidence must be a number"),
    ],
)
def test_priority_request_rejects_malformed_fields(kwargs, match):
    base = dict(approach="Road 1", active=True, reason="test", timestamp=1.0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        PriorityRequest(**base)


def test_priority_request_accepts_missing_confidence_and_empty_evidence():
    request = PriorityRequest(approach="Road 1", active=False, reason="no evidence")

    assert request.confidence is None
    assert request.event_ids == ()
    assert request.to_dict()["confidence"] is None


def test_detection_event_accepts_zero_window_boundary():
    event = DetectionEvent.audio(
        "Road 1", 0.5, window_start_s=0.0, window_end_s=0.1, timestamp=0.0
    )

    assert event.window_start_s == 0.0
    assert event.status is EventStatus.CANDIDATE
    assert utc_iso(event.timestamp).startswith("1970-01-01T00:00:00")
