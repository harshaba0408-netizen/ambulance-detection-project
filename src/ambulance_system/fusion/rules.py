"""Pure fusion policy rules.

Confidence values from audio and video are not commensurable. They are never
averaged. A policy either selects one modality's confidence or, for conjunction
policies, reports the **minimum** of the two (the conservative bound implied by
requiring both).

The engine filters events by `fusion.stale_after_s` before calling these
functions. `ADAPTIVE` additionally applies `fusion.adaptive_window_s`, which is
validated to be no larger than the staleness window. ADAPTIVE is therefore
never looser than BOTH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ambulance_system.config import VALID_POLICIES
from ambulance_system.events import DetectionEvent, EventSource

POLICY_VIDEO = "VIDEO"
POLICY_AUDIO = "AUDIO"
POLICY_BOTH = "BOTH"
POLICY_ADAPTIVE = "ADAPTIVE"


@dataclass(frozen=True)
class PolicyResult:
    """Outcome of applying one policy to one approach's fresh evidence."""

    active: bool
    confidence: Optional[float]
    reason: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool):
            raise ValueError("active must be a bool")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be within [0.0, 1.0] or None")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not isinstance(self.evidence_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence_ids
        ):
            raise ValueError("evidence_ids must be a tuple of non-empty strings")


def is_fresh(event: DetectionEvent, now: float, max_age_s: float) -> bool:
    """True when the event is no older than `max_age_s` at `now`."""
    if not isinstance(event, DetectionEvent):
        raise TypeError("event must be a DetectionEvent")
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not float(now) >= 0.0:
        raise ValueError("now must be a number >= 0")
    if isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float)) or float(max_age_s) <= 0.0:
        raise ValueError("max_age_s must be > 0")
    return float(now) - float(event.timestamp) <= float(max_age_s)


def _split(events: Iterable[DetectionEvent]) -> tuple[list[DetectionEvent], list[DetectionEvent]]:
    video: list[DetectionEvent] = []
    audio: list[DetectionEvent] = []
    for event in events:
        if not isinstance(event, DetectionEvent):
            raise TypeError("events must contain DetectionEvent objects")
        if event.source is EventSource.VIDEO:
            video.append(event)
        elif event.source is EventSource.AUDIO:
            audio.append(event)
    return video, audio


def _max_confidence(events: list[DetectionEvent]) -> Optional[float]:
    if not events:
        return None
    return max(float(event.confidence) for event in events)


def _ids(*groups: list[DetectionEvent]) -> tuple[str, ...]:
    return tuple(sorted(event.event_id for group in groups for event in group))


def evaluate_policy(
    policy: str,
    events: Iterable[DetectionEvent],
    *,
    now: float,
    adaptive_window_s: float = 1.0,
) -> PolicyResult:
    """Apply one policy to the events already filtered for staleness."""
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown fusion policy: {policy!r}")

    video, audio = _split(events)
    if policy == POLICY_ADAPTIVE:
        if float(adaptive_window_s) <= 0.0:
            raise ValueError("adaptive_window_s must be > 0")
        video = [event for event in video if is_fresh(event, now, adaptive_window_s)]
        audio = [event for event in audio if is_fresh(event, now, adaptive_window_s)]

    video_confidence = _max_confidence(video)
    audio_confidence = _max_confidence(audio)

    if policy == POLICY_VIDEO:
        if video:
            return PolicyResult(True, video_confidence, "fresh video evidence", _ids(video))
        return PolicyResult(False, None, "no fresh video evidence", ())

    if policy == POLICY_AUDIO:
        if audio:
            return PolicyResult(True, audio_confidence, "fresh audio evidence", _ids(audio))
        return PolicyResult(False, None, "no fresh audio evidence", ())

    # BOTH and ADAPTIVE are conjunctions.
    if policy == POLICY_ADAPTIVE:
        active = bool(video and audio)
        if active:
            return PolicyResult(
                True,
                min(video_confidence, audio_confidence),
                "fresh video and audio evidence within the adaptive window",
                _ids(video, audio),
            )
        if not video and not audio:
            return PolicyResult(
                False,
                None,
                "no fresh video or audio evidence within the adaptive window",
                (),
            )
        if not video:
            return PolicyResult(
                False, None, "missing fresh video evidence within the adaptive window", ()
            )
        return PolicyResult(
            False, None, "missing fresh audio evidence within the adaptive window", ()
        )

    # POLICY_BOTH
    if video and audio:
        return PolicyResult(
            True,
            min(video_confidence, audio_confidence),
            "fresh video and audio evidence",
            _ids(video, audio),
        )
    if not video and not audio:
        return PolicyResult(False, None, "no fresh video or audio evidence", ())
    if not video:
        return PolicyResult(False, None, "missing fresh video evidence", ())
    return PolicyResult(False, None, "missing fresh audio evidence", ())
