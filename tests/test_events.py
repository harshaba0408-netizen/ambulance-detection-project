from __future__ import annotations

import pytest

from ambulance_system.events import (
    DetectionEvent,
    DetectionType,
    EventSource,
    EventStatus,
    PriorityRequest,
    new_event_id,
    utc_iso,
)


def test_new_event_ids_are_unique():
    ids = {new_event_id() for _ in range(200)}

    assert len(ids) == 200
    assert all(len(event_id) == 32 for event_id in ids)


def test_video_event_factory_sets_modality_fields():
    event = DetectionEvent.video(
        "Road 1",
        0.87,
        bbox=(10.0, 20.0, 110.0, 120.0),
        frame_index=42,
        timestamp=1234.5,
    )

    assert event.source is EventSource.VIDEO
    assert event.detection_type is DetectionType.AMBULANCE
    assert event.status is EventStatus.CANDIDATE
    assert event.frame_index == 42
    assert event.timestamp == pytest.approx(1234.5)


def test_audio_event_factory_sets_window_fields():
    event = DetectionEvent.audio(
        "Road 2",
        0.91,
        window_start_s=1.0,
        window_end_s=4.0,
        timestamp=10.0,
    )

    assert event.source is EventSource.AUDIO
    assert event.detection_type is DetectionType.SIREN
    assert event.window_start_s == pytest.approx(1.0)
    assert event.window_end_s == pytest.approx(4.0)


def test_event_serialises_to_plain_dict():
    event = DetectionEvent.video("Road 1", 0.75, bbox=(1, 2, 3, 4), timestamp=0.0)
    payload = event.to_dict()

    assert payload["source"] == "video"
    assert payload["detection_type"] == "ambulance"
    assert payload["status"] == "candidate"
    assert payload["bbox"] == [1, 2, 3, 4]
    assert payload["timestamp_iso"].startswith("1970-01-01T00:00:00")
    assert payload["timestamp_iso"].endswith("+00:00")


def test_event_round_trips_through_dict():
    event = DetectionEvent.audio(
        "Road 3",
        0.66,
        window_start_s=2.0,
        window_end_s=5.0,
        timestamp=100.0,
        evidence_refs=("evt-1", "evt-2"),
    )
    rebuilt = DetectionEvent.from_dict(event.to_dict())

    assert rebuilt == event


def test_confidence_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        DetectionEvent.video("Road 1", 1.01)
    with pytest.raises(ValueError, match="confidence"):
        DetectionEvent.audio("Road 1", -0.01)
    with pytest.raises(ValueError, match="confidence"):
        DetectionEvent.video("Road 1", float("nan"))


def test_empty_approach_is_rejected():
    with pytest.raises(ValueError, match="approach"):
        DetectionEvent.video("   ", 0.5)


def test_negative_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timestamp"):
        DetectionEvent.video("Road 1", 0.5, timestamp=-1.0)


def test_inverted_bbox_is_rejected():
    with pytest.raises(ValueError, match="bbox"):
        DetectionEvent.video("Road 1", 0.5, bbox=(100.0, 0.0, 10.0, 50.0))


def test_partial_audio_window_is_rejected():
    with pytest.raises(ValueError, match="window_start_s and window_end_s"):
        DetectionEvent.audio("Road 1", 0.5, window_start_s=1.0)


def test_inverted_audio_window_is_rejected():
    with pytest.raises(ValueError, match="window_end_s"):
        DetectionEvent.audio("Road 1", 0.5, window_start_s=4.0, window_end_s=4.0)


def test_with_status_returns_a_new_event():
    event = DetectionEvent.video("Road 1", 0.9, timestamp=5.0)
    confirmed = event.with_status(EventStatus.CONFIRMED)

    assert confirmed is not event
    assert confirmed.status is EventStatus.CONFIRMED
    assert event.status is EventStatus.CANDIDATE
    assert confirmed.event_id == event.event_id


def test_priority_request_serialises():
    request = PriorityRequest(
        approach="Road 1",
        active=True,
        reason="video+audio confirmed",
        timestamp=20.0,
        event_ids=("a", "b"),
        confidence=0.8,
    )
    payload = request.to_dict()

    assert payload["active"] is True
    assert payload["event_ids"] == ["a", "b"]
    assert payload["confidence"] == pytest.approx(0.8)


def test_priority_request_rejects_non_boolean_active():
    with pytest.raises(ValueError, match="active"):
        PriorityRequest(approach="Road 1", active=1, reason="test")


def test_priority_request_requires_reason():
    with pytest.raises(ValueError, match="reason"):
        PriorityRequest(approach="Road 1", active=True, reason="")


def test_utc_iso_format():
    assert utc_iso(0.0) == "1970-01-01T00:00:00+00:00"
