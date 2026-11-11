from __future__ import annotations

from dataclasses import replace

import pytest

from ambulance_system.clock import FakeClock
from ambulance_system.config import SignalConfig
from ambulance_system.traffic_signal import (
    ApproachObservation,
    PriorityPhase,
    SafetyViolationError,
    SignalColor,
    SignalController,
)

APPROACHES = ("Road 1", "Road 2", "Road 3", "Road 4")

BASE_SETTINGS = {
    "activation_threshold": 0.7,
    "release_threshold": 0.4,
    "min_confirmation_s": 1.0,
    "release_grace_s": 2.0,
    "max_priority_s": 10.0,
    "min_green_s": 2.0,
    "yellow_s": 1.0,
    "all_red_s": 1.0,
}


def controller(**overrides) -> SignalController:
    settings = dict(BASE_SETTINGS)
    settings.update(overrides)
    return SignalController(
        replace(SignalConfig(), **settings),
        APPROACHES,
        clock=FakeClock(start=0.0),
    )


def obs(approach: str, confidence: float = 1.0, requested: bool = True) -> ApproachObservation:
    return ApproachObservation(approach, requested, confidence)


def assert_safe(signal: SignalController) -> None:
    states = signal.states.values()
    greens = [state.approach for state in states if state.colour is SignalColor.GREEN]
    yellows = [state.approach for state in states if state.colour is SignalColor.YELLOW]
    assert len(greens) <= 1
    assert not (greens and yellows)


# ---------------------------------------------------------------------------
# Initial state and grant
# ---------------------------------------------------------------------------
def test_initial_state_is_all_red_normal():
    signal = controller()

    assert all(state.phase is PriorityPhase.NORMAL for state in signal.states.values())
    assert all(state.colour is SignalColor.RED for state in signal.states.values())
    assert signal.active_approach is None
    assert signal.pending_approaches == ()


def test_grant_waits_for_confirmation_and_all_red_interval():
    signal = controller()

    assert signal.step({"Road 1": obs("Road 1")}, now=0.0) == []
    assert signal.step({"Road 1": obs("Road 1")}, now=0.999) == []
    assert signal.active_approach is None

    commands = signal.step({"Road 1": obs("Road 1")}, now=1.0)

    assert len(commands) == 1
    assert commands[0].approach == "Road 1"
    assert commands[0].colour is SignalColor.GREEN
    assert signal.active_approach == "Road 1"
    assert signal.states["Road 2"].colour is SignalColor.RED
    assert_safe(signal)


def test_initial_green_respects_all_red_interval():
    signal = controller(min_confirmation_s=0.0, all_red_s=3.0)

    assert signal.step({"Road 1": obs("Road 1")}, now=0.0) == []
    assert signal.step({"Road 1": obs("Road 1")}, now=2.999) == []
    commands = signal.step({"Road 1": obs("Road 1")}, now=3.0)

    assert len(commands) == 1
    assert signal.active_approach == "Road 1"


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------
def test_earliest_confirmed_request_wins():
    signal = controller()

    signal.step({"Road 1": obs("Road 1")}, now=0.0)
    # Road 2 evidence only starts at 0.2, so its confirmation completes later.
    signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2")}, now=0.2)
    commands = signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2")}, now=1.0)

    assert len(commands) == 1
    assert signal.active_approach == "Road 1"
    assert signal.pending_approaches == ()

    signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2")}, now=1.2)
    assert signal.pending_approaches == ("Road 2",)


def test_tie_is_broken_by_configured_order():
    signal = controller()

    signal.step({"Road 3": obs("Road 3"), "Road 2": obs("Road 2")}, now=0.0)
    signal.step({"Road 3": obs("Road 3"), "Road 2": obs("Road 2")}, now=1.0)

    assert signal.active_approach == "Road 2"


def test_active_priority_is_not_preempted_by_a_later_request():
    signal = controller()

    signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2")}, now=0.0)
    signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2")}, now=1.0)
    signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2")}, now=2.0)

    assert signal.active_approach == "Road 1"
    assert signal.states["Road 2"].colour is SignalColor.RED
    assert signal.pending_approaches == ("Road 2",)
    assert_safe(signal)


# ---------------------------------------------------------------------------
# Release and handover
# ---------------------------------------------------------------------------
def test_release_sequence_returns_to_all_red():
    signal = controller()

    for moment in (0.0, 1.0):
        signal.step({"Road 1": obs("Road 1")}, now=moment)

    signal.step({"Road 1": obs("Road 1", confidence=0.2)}, now=2.0)
    signal.step({"Road 1": obs("Road 1", confidence=0.2)}, now=3.0)
    commands = signal.step({"Road 1": obs("Road 1", confidence=0.2)}, now=4.0)

    assert any(command.colour is SignalColor.YELLOW for command in commands)
    assert signal.states["Road 1"].phase is PriorityPhase.CLEARING
    assert signal.active_approach is None

    signal.step({"Road 1": obs("Road 1", confidence=0.2)}, now=5.0)
    assert signal.states["Road 1"].colour is SignalColor.RED

    signal.step({"Road 1": obs("Road 1", confidence=0.2)}, now=6.0)
    assert signal.states["Road 1"].phase is PriorityPhase.NORMAL
    assert signal.states["Road 1"].colour is SignalColor.RED
    assert_safe(signal)


def test_handover_never_shows_two_greens_or_green_with_yellow():
    signal = controller()

    def observations(moment: float) -> dict[str, ApproachObservation]:
        road_1_confidence = 1.0 if moment < 2.0 else 0.2
        return {
            "Road 1": obs("Road 1", road_1_confidence),
            "Road 2": obs("Road 2"),
        }

    for moment in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        signal.step(observations(moment), now=moment)
        assert_safe(signal)

    road_1 = signal.states["Road 1"]
    road_2 = signal.states["Road 2"]

    assert road_1.phase is PriorityPhase.NORMAL
    assert road_1.colour is SignalColor.RED
    assert road_2.phase is PriorityPhase.EMERGENCY_PRIORITY
    assert road_2.colour is SignalColor.GREEN
    assert signal.active_approach == "Road 2"


def test_handover_waits_for_all_red_after_yellow():
    signal = controller()

    def observations(moment: float) -> dict[str, ApproachObservation]:
        return {
            "Road 1": obs("Road 1", 1.0 if moment < 2.0 else 0.2),
            "Road 2": obs("Road 2"),
        }

    for moment in (0.0, 1.0, 2.0, 3.0, 4.0):
        signal.step(observations(moment), now=moment)

    # Road 1 is now yellow; Road 2 must not be green yet.
    assert signal.states["Road 1"].colour is SignalColor.YELLOW
    assert signal.states["Road 2"].colour is SignalColor.RED

    signal.step(observations(5.0), now=5.0)
    assert signal.states["Road 1"].colour is SignalColor.RED
    assert signal.states["Road 2"].colour is SignalColor.RED

    signal.step(observations(5.5), now=5.5)
    assert signal.states["Road 2"].colour is SignalColor.RED  # all-red still running

    signal.step(observations(6.0), now=6.0)
    assert signal.states["Road 2"].colour is SignalColor.GREEN


def test_hysteresis_prevents_release_between_thresholds():
    signal = controller()

    for moment in (0.0, 1.0):
        signal.step({"Road 1": obs("Road 1")}, now=moment)

    for moment in (2.0, 3.0, 4.0, 5.0):
        signal.step({"Road 1": obs("Road 1", confidence=0.5)}, now=moment)

    assert signal.active_approach == "Road 1"
    assert signal.states["Road 1"].colour is SignalColor.GREEN


def test_missing_observation_counts_as_no_evidence():
    signal = controller()

    for moment in (0.0, 1.0):
        signal.step({"Road 1": obs("Road 1")}, now=moment)

    signal.step({}, now=2.0)
    signal.step({}, now=3.0)
    signal.step({}, now=4.0)

    assert signal.states["Road 1"].phase is PriorityPhase.CLEARING
    assert signal.states["Road 1"].colour is SignalColor.YELLOW


def test_maximum_priority_duration_releases_even_with_strong_evidence():
    signal = controller(max_priority_s=10.0)

    signal.step({"Road 1": obs("Road 1")}, now=0.0)
    signal.step({"Road 1": obs("Road 1")}, now=1.0)
    assert signal.active_approach == "Road 1"

    signal.step({"Road 1": obs("Road 1")}, now=10.9)
    assert signal.active_approach == "Road 1"

    commands = signal.step({"Road 1": obs("Road 1")}, now=11.0)

    assert any(command.phase is PriorityPhase.CLEARING for command in commands)
    assert signal.states["Road 1"].colour is SignalColor.YELLOW


# ---------------------------------------------------------------------------
# Safety actions
# ---------------------------------------------------------------------------
def test_clear_all_forces_every_approach_to_red():
    signal = controller()
    signal.step({"Road 1": obs("Road 1")}, now=0.0)
    signal.step({"Road 1": obs("Road 1")}, now=1.0)
    assert signal.active_approach == "Road 1"

    commands = signal.clear_all(now=2.0)

    assert len(commands) == 1
    assert all(state.phase is PriorityPhase.NORMAL for state in signal.states.values())
    assert all(state.colour is SignalColor.RED for state in signal.states.values())
    assert signal.active_approach is None
    assert signal.pending_approaches == ()


def test_fault_is_all_red_and_blocks_decisions_until_recovery():
    signal = controller()
    signal.step({"Road 1": obs("Road 1")}, now=0.0)
    signal.step({"Road 1": obs("Road 1")}, now=1.0)

    signal.trigger_fault("camera failure", now=2.0)

    assert signal.is_fault
    assert all(state.phase is PriorityPhase.FAULT for state in signal.states.values())
    assert all(state.colour is SignalColor.RED for state in signal.states.values())

    assert signal.step({"Road 1": obs("Road 1")}, now=3.0) == []
    assert signal.is_fault

    signal.clear_fault(now=4.0)

    assert not signal.is_fault
    assert all(state.phase is PriorityPhase.NORMAL for state in signal.states.values())

    signal.step({"Road 1": obs("Road 1")}, now=5.0)
    signal.step({"Road 1": obs("Road 1")}, now=6.0)
    assert signal.active_approach == "Road 1"


def test_clear_fault_is_noop_when_not_in_fault():
    signal = controller()

    assert signal.clear_fault(now=0.0) == []


def test_pending_request_is_dropped_if_evidence_disappears():
    signal = controller()

    signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2")}, now=0.0)
    signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2")}, now=1.0)

    assert signal.pending_approaches == ("Road 2",)

    signal.step({"Road 1": obs("Road 1"), "Road 2": obs("Road 2", requested=False)}, now=2.0)

    assert signal.pending_approaches == ()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_unknown_approach_is_rejected():
    signal = controller()

    with pytest.raises(ValueError, match="unknown approach"):
        signal.step({"Road 9": obs("Road 9")}, now=0.0)


def test_mismatched_observation_key_is_rejected():
    signal = controller()

    with pytest.raises(ValueError, match="does not match"):
        signal.step({"Road 1": obs("Road 2")}, now=0.0)


def test_time_cannot_move_backwards():
    signal = controller()
    signal.step(now=5.0)

    with pytest.raises(ValueError, match="backwards"):
        signal.step(now=4.0)


def test_invalid_observation_confidence_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        obs("Road 1", confidence=1.5)


def test_safety_error_type_is_exposed():
    assert issubclass(SafetyViolationError, RuntimeError)
