from __future__ import annotations

import pytest

from ambulance_system.clock import FakeClock
from ambulance_system.config import FusionConfig, SignalConfig
from ambulance_system.events import DetectionEvent
from ambulance_system.fusion import (
    FusionDecision,
    FusionEngine,
    POLICY_ADAPTIVE,
    POLICY_AUDIO,
    POLICY_BOTH,
    POLICY_VIDEO,
    PolicyResult,
    evaluate_policy,
    is_fresh,
)
from ambulance_system.traffic_signal import SignalController


def video(
    approach: str = "Road 1",
    confidence: float = 0.9,
    timestamp: float = 0.0,
    event_id: str = "v1",
) -> DetectionEvent:
    return DetectionEvent.video(
        approach=approach,
        confidence=confidence,
        timestamp=timestamp,
        event_id=event_id,
    )


def audio(
    approach: str = "Road 1",
    confidence: float = 0.6,
    timestamp: float = 0.0,
    event_id: str = "a1",
) -> DetectionEvent:
    return DetectionEvent.audio(
        approach=approach,
        confidence=confidence,
        window_start_s=0.0,
        window_end_s=3.0,
        timestamp=timestamp,
        event_id=event_id,
    )


def config(policy: str, *, stale: float = 3.0, debounce: float = 1.0, adaptive: float = 1.0) -> FusionConfig:
    return FusionConfig(
        policy=policy,
        stale_after_s=stale,
        debounce_s=debounce,
        adaptive_window_s=adaptive,
    )


def engine(policy: str, approaches=("Road 1",), **kwargs) -> FusionEngine:
    return FusionEngine(config(policy, **kwargs), list(approaches), clock=FakeClock(0.0))


# ---------------------------------------------------------------------------
# Policy truth table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "policy, has_video, has_audio, expected_active, expected_confidence",
    [
        (POLICY_VIDEO, True, False, True, 0.9),
        (POLICY_VIDEO, False, True, False, None),
        (POLICY_VIDEO, True, True, True, 0.9),
        (POLICY_VIDEO, False, False, False, None),
        (POLICY_AUDIO, True, False, False, None),
        (POLICY_AUDIO, False, True, True, 0.6),
        (POLICY_AUDIO, True, True, True, 0.6),
        (POLICY_AUDIO, False, False, False, None),
        (POLICY_BOTH, True, False, False, None),
        (POLICY_BOTH, False, True, False, None),
        (POLICY_BOTH, True, True, True, 0.6),
        (POLICY_BOTH, False, False, False, None),
        (POLICY_ADAPTIVE, True, False, False, None),
        (POLICY_ADAPTIVE, False, True, False, None),
        (POLICY_ADAPTIVE, True, True, True, 0.6),
        (POLICY_ADAPTIVE, False, False, False, None),
    ],
)
def test_policy_truth_table(policy, has_video, has_audio, expected_active, expected_confidence):
    fusion = engine(policy)
    if has_video:
        assert fusion.submit(video(confidence=0.9, event_id="v1")) is True
    if has_audio:
        assert fusion.submit(audio(confidence=0.6, event_id="a1")) is True

    update = fusion.evaluate(now=0.0)
    decision = update.decisions["Road 1"]

    assert decision.active is expected_active
    if expected_confidence is None:
        assert decision.confidence is None
    else:
        assert decision.confidence == pytest.approx(expected_confidence)

    if policy == POLICY_VIDEO:
        expected_ids = ("v1",) if has_video else ()
    elif policy == POLICY_AUDIO:
        expected_ids = ("a1",) if has_audio else ()
    elif has_video and has_audio:
        expected_ids = ("a1", "v1")
    else:
        expected_ids = ()
    assert decision.evidence_ids == expected_ids


def test_both_policy_never_averages_confidences():
    fusion = engine(POLICY_BOTH)
    fusion.submit(video(confidence=0.9, event_id="v1"))
    fusion.submit(audio(confidence=0.2, event_id="a1"))

    decision = fusion.evaluate(now=0.0).decisions["Road 1"]

    assert decision.confidence == pytest.approx(0.2)
    assert decision.confidence != pytest.approx((0.9 + 0.2) / 2)


def test_conflicting_simultaneous_requests_are_arbitrated_by_the_controller():
    approaches = ["Road 1", "Road 2"]
    fusion = FusionEngine(config(POLICY_VIDEO), approaches, clock=FakeClock(0.0))
    fusion.submit(video("Road 1", confidence=0.9, event_id="v1"))
    fusion.submit(video("Road 2", confidence=0.95, event_id="v2"))

    update = fusion.evaluate(now=0.0)
    assert update.decisions["Road 1"].active is True
    assert update.decisions["Road 2"].active is True

    controller = SignalController(
        SignalConfig(min_confirmation_s=0.0), approaches, clock=FakeClock(0.0)
    )
    commands = controller.step(update.to_observations(), now=2.0)

    assert len(commands) == 1
    assert controller.active_approach == "Road 1"
    assert controller.states["Road 2"].colour.value == "red"


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
def test_evidence_expires_after_the_staleness_window():
    fusion = engine(POLICY_VIDEO, stale=3.0)
    fusion.submit(video(timestamp=10.0))

    first = fusion.evaluate(now=10.0)
    assert first.decisions["Road 1"].active is True
    assert len(first.changed) == 1

    still_fresh = fusion.evaluate(now=13.0)
    assert still_fresh.decisions["Road 1"].active is True
    assert still_fresh.changed == ()

    expired = fusion.evaluate(now=13.01)
    assert expired.decisions["Road 1"].active is False
    assert len(expired.changed) == 1
    assert expired.changed[0].evidence_ids == ()
    assert fusion.stats.expired == 1


def test_partial_expiry_keeps_the_other_modality():
    fusion = engine(POLICY_BOTH, stale=3.0)
    fusion.submit(video(timestamp=0.0, event_id="v1"))
    fusion.submit(audio(timestamp=2.5, event_id="a1"))

    assert fusion.evaluate(now=3.0).decisions["Road 1"].active is True

    update = fusion.evaluate(now=3.01)
    decision = update.decisions["Road 1"]

    assert decision.active is False
    assert "video" in decision.reason


def test_adaptive_window_is_stricter_than_both():
    both = engine(POLICY_BOTH, stale=3.0, adaptive=1.0)
    adaptive = engine(POLICY_ADAPTIVE, stale=3.0, adaptive=1.0)
    for fusion in (both, adaptive):
        fusion.submit(video(timestamp=0.0, event_id="v1"))
        fusion.submit(audio(timestamp=2.0, event_id="a1"))

    both_update = both.evaluate(now=2.0)
    adaptive_update = adaptive.evaluate(now=2.0)

    assert both_update.decisions["Road 1"].active is True
    assert adaptive_update.decisions["Road 1"].active is False
    assert "adaptive window" in adaptive_update.decisions["Road 1"].reason


# ---------------------------------------------------------------------------
# Duplicate suppression
# ---------------------------------------------------------------------------
def test_duplicate_suppression_uses_key_and_debounce_window():
    fusion = engine(POLICY_VIDEO, debounce=1.0)

    assert fusion.submit(video(timestamp=10.0)) is True
    assert fusion.submit(video(timestamp=10.5)) is False
    assert fusion.submit(video(timestamp=11.0)) is True

    assert fusion.stats.accepted == 2
    assert fusion.stats.suppressed == 1


def test_different_approach_or_source_is_not_a_duplicate():
    fusion = engine(POLICY_VIDEO, approaches=("Road 1", "Road 2"), debounce=1.0)

    assert fusion.submit(video("Road 1", timestamp=10.0, event_id="v1")) is True
    assert fusion.submit(video("Road 2", timestamp=10.0, event_id="v2")) is True
    assert fusion.submit(audio("Road 1", timestamp=10.0, event_id="a1")) is True

    assert fusion.stats.accepted == 3
    assert fusion.stats.suppressed == 0


def test_out_of_order_duplicate_is_suppressed():
    fusion = engine(POLICY_VIDEO, debounce=1.0)
    fusion.submit(video(timestamp=11.0))

    assert fusion.submit(video(timestamp=9.0)) is False


def test_multiple_fresh_events_use_max_confidence_and_all_ids():
    fusion = engine(POLICY_BOTH, debounce=1.0)
    fusion.submit(video(confidence=0.7, timestamp=0.0, event_id="v1"))
    fusion.submit(video(confidence=0.9, timestamp=2.0, event_id="v2"))
    fusion.submit(audio(confidence=0.5, timestamp=2.0, event_id="a1"))

    decision = fusion.evaluate(now=2.0).decisions["Road 1"]

    assert decision.active is True
    assert decision.confidence == pytest.approx(0.5)
    assert decision.evidence_ids == ("a1", "v1", "v2")


# ---------------------------------------------------------------------------
# Validation and conversion
# ---------------------------------------------------------------------------
def test_unknown_approach_and_wrong_type_are_rejected():
    fusion = engine(POLICY_VIDEO)

    with pytest.raises(ValueError, match="unknown approach"):
        fusion.submit(video("Road 9"))
    with pytest.raises(TypeError, match="DetectionEvent"):
        fusion.submit(object())


def test_time_cannot_move_backwards():
    fusion = engine(POLICY_VIDEO)
    fusion.evaluate(now=5.0)

    with pytest.raises(ValueError, match="backwards"):
        fusion.evaluate(now=4.0)


def test_constructor_validates_approaches():
    with pytest.raises(ValueError, match="at least one"):
        FusionEngine(config(POLICY_VIDEO), [])
    with pytest.raises(ValueError, match="unique"):
        FusionEngine(config(POLICY_VIDEO), ["Road 1", "Road 1"])
    with pytest.raises(ValueError, match="non-empty"):
        FusionEngine(config(POLICY_VIDEO), ["  "])


def test_decisions_convert_to_controller_observations_and_priority_requests():
    fusion = engine(POLICY_VIDEO)
    fusion.submit(video(confidence=0.87, event_id="v1"))
    decision = fusion.evaluate(now=0.0).decisions["Road 1"]

    observation = decision.to_observation()
    request = decision.to_priority_request()

    assert observation.approach == "Road 1"
    assert observation.requested is True
    assert observation.confidence == pytest.approx(0.87)
    assert request.active is True
    assert request.confidence == pytest.approx(0.87)
    assert request.event_ids == ("v1",)
    assert request.reason == "fresh video evidence"


def test_inactive_decision_uses_zero_confidence_for_the_controller():
    fusion = engine(POLICY_VIDEO)
    decision = fusion.evaluate(now=0.0).decisions["Road 1"]

    assert decision.active is False
    assert decision.confidence is None
    assert decision.to_observation().confidence == 0.0
    assert decision.to_priority_request().confidence is None


def test_decision_and_policy_result_validate_fields():
    with pytest.raises(ValueError, match="active"):
        FusionDecision("Road 1", 1, None, "reason", (), 0.0)
    with pytest.raises(ValueError, match="confidence"):
        FusionDecision("Road 1", True, 1.5, "reason", (), 0.0)
    with pytest.raises(ValueError, match="reason"):
        FusionDecision("Road 1", True, 0.5, " ", (), 0.0)
    with pytest.raises(ValueError, match="evidence_ids"):
        FusionDecision("Road 1", True, 0.5, "reason", ("",), 0.0)

    with pytest.raises(ValueError, match="active"):
        PolicyResult(1, 0.5, "reason", ())
    with pytest.raises(ValueError, match="confidence"):
        PolicyResult(True, 2.0, "reason", ())
    with pytest.raises(ValueError, match="reason"):
        PolicyResult(True, 0.5, "", ())


def test_rules_reject_unknown_policy_and_bad_events():
    with pytest.raises(ValueError, match="unknown fusion policy"):
        evaluate_policy("MAGIC", [], now=0.0)
    with pytest.raises(TypeError, match="DetectionEvent"):
        evaluate_policy(POLICY_VIDEO, [object()], now=0.0)


def test_is_fresh_checks_the_boundary():
    event = video(timestamp=10.0)

    assert is_fresh(event, now=10.0, max_age_s=3.0) is True
    assert is_fresh(event, now=13.0, max_age_s=3.0) is True
    assert is_fresh(event, now=13.01, max_age_s=3.0) is False

    with pytest.raises(ValueError, match="max_age_s"):
        is_fresh(event, now=10.0, max_age_s=0)
    with pytest.raises(TypeError, match="DetectionEvent"):
        is_fresh(object(), now=10.0, max_age_s=1.0)


def test_active_approaches_and_update_serialisation():
    fusion = engine(POLICY_VIDEO, approaches=("Road 1", "Road 2"))
    fusion.submit(video("Road 1", event_id="v1"))
    update = fusion.evaluate(now=0.0)

    assert fusion.active_approaches == ("Road 1",)
    payload = update.to_dict()
    assert len(payload["decisions"]) == 2
    assert len(payload["changed"]) == 1
    assert payload["changed"][0]["evidence_ids"] == ["v1"]


def test_per_approach_policy_overrides_and_validation():
    fusion = FusionEngine(
        config(POLICY_AUDIO),
        ["Road 1", "Road 2"],
        policies={"Road 2": POLICY_VIDEO},
        clock=FakeClock(0.0),
    )
    assert fusion.policy == POLICY_AUDIO
    assert fusion.policies == {"Road 1": POLICY_AUDIO, "Road 2": POLICY_VIDEO}
    assert fusion.policy_for("Road 2") == POLICY_VIDEO

    fusion.submit(video("Road 2", event_id="v1"))
    update = fusion.evaluate(now=0.0)
    assert update.decisions["Road 1"].active is False
    assert update.decisions["Road 2"].active is True

    fusion.set_policy("Road 1", POLICY_VIDEO)
    assert fusion.policy_for("Road 1") == POLICY_VIDEO
    assert fusion.evaluate(now=0.5).decisions["Road 1"].active is False

    with pytest.raises(ValueError, match="unknown fusion policy"):
        fusion.set_policy("Road 1", "MAGIC")
    with pytest.raises(ValueError, match="unknown approach"):
        fusion.set_policy("Road 9", POLICY_VIDEO)
    with pytest.raises(ValueError, match="unknown approach"):
        FusionEngine(config(POLICY_AUDIO), ["Road 1"], policies={"Road 2": POLICY_VIDEO})
    with pytest.raises(ValueError, match="unknown fusion policy"):
        FusionEngine(config(POLICY_AUDIO), ["Road 1"], policies={"Road 1": "MAGIC"})


def test_reset_discards_evidence_and_active_flags_but_not_counters_or_time():
    fusion = engine(POLICY_VIDEO)
    fusion.submit(video("Road 1", timestamp=0.0))
    assert fusion.evaluate(now=0.0).decisions["Road 1"].active is True
    assert fusion.stats.accepted == 1

    fusion.reset()
    assert fusion.active_approaches == ()
    update = fusion.evaluate(now=0.5)
    assert update.decisions["Road 1"].active is False
    assert fusion.stats.accepted == 1  # operational counters are retained

    with pytest.raises(ValueError, match="backwards"):
        fusion.evaluate(now=0.4)

