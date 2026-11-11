from __future__ import annotations

from dataclasses import replace

import pytest

from ambulance_system.config import SignalConfig
from ambulance_system.traffic_signal import (
    ApproachMachine,
    ApproachObservation,
    ApproachSnapshot,
    InvalidTransitionError,
    PriorityPhase,
    SignalColor,
    SignalCommand,
    validate_transition,
)


def config(**overrides) -> SignalConfig:
    return replace(SignalConfig(), **overrides)


def observation(confidence: float = 1.0, requested: bool = True) -> ApproachObservation:
    return ApproachObservation("Road 1", requested, confidence)


def machine(**overrides) -> ApproachMachine:
    return ApproachMachine("Road 1", config(**overrides), now=0.0)


# ---------------------------------------------------------------------------
# Initial state and confirmation
# ---------------------------------------------------------------------------
def test_initial_state_is_normal_red():
    unit = machine()

    assert unit.phase is PriorityPhase.NORMAL
    assert unit.colour is SignalColor.RED
    assert unit.priority_started_at is None


def test_request_requires_full_confirmation_duration():
    unit = machine(min_confirmation_s=1.0)

    assert unit.observe(observation(), 0.0) is None
    assert unit.observe(observation(), 0.5) is None
    assert unit.observe(observation(), 0.999) is None
    assert unit.observe(observation(), 1.0) == pytest.approx(1.0)


def test_low_confidence_restarts_confirmation():
    unit = machine(min_confirmation_s=1.0, activation_threshold=0.7)

    assert unit.observe(observation(0.6), 0.0) is None
    assert unit.observe(observation(0.6), 0.5) is None
    assert unit.observe(observation(0.8), 1.0) is None
    assert unit.observe(observation(0.8), 1.9) is None
    assert unit.observe(observation(0.8), 2.0) == pytest.approx(2.0)


def test_not_requested_resets_confirmation():
    unit = machine(min_confirmation_s=1.0)

    assert unit.observe(observation(1.0), 0.0) is None
    assert unit.observe(observation(1.0, requested=False), 0.5) is None
    assert unit.observe(observation(1.0), 1.0) is None
    assert unit.observe(observation(1.0), 2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Activation and hysteresis
# ---------------------------------------------------------------------------
def test_activation_emits_green_command():
    unit = machine()
    command = unit.activate(5.0, "priority confirmed")

    assert command.approach == "Road 1"
    assert command.colour is SignalColor.GREEN
    assert command.phase is PriorityPhase.EMERGENCY_PRIORITY
    assert command.timestamp == pytest.approx(5.0)
    assert unit.phase is PriorityPhase.EMERGENCY_PRIORITY
    assert unit.colour is SignalColor.GREEN
    assert unit.priority_started_at == pytest.approx(5.0)


def test_duplicate_activation_is_rejected():
    unit = machine()
    unit.activate(0.0)

    with pytest.raises(InvalidTransitionError, match="cannot activate"):
        unit.activate(1.0)


def test_clearing_cannot_start_from_normal():
    unit = machine()

    with pytest.raises(InvalidTransitionError, match="cannot begin clearing"):
        unit.begin_clearing(0.0)


def test_hysteresis_keeps_priority_between_thresholds():
    unit = machine(activation_threshold=0.7, release_threshold=0.4)
    unit.activate(0.0)

    assert unit.observe(observation(0.5), 1.0) is None
    assert unit.release_due(10.0) is False
    assert unit.colour is SignalColor.GREEN


def test_release_requires_grace_period():
    unit = machine(min_green_s=0.1, release_grace_s=2.0)
    unit.activate(0.0)
    unit.observe(observation(0.2), 1.0)

    assert unit.release_due(2.999) is False
    assert unit.release_due(3.0) is True


def test_minimum_green_blocks_release():
    unit = machine(min_green_s=5.0, release_grace_s=1.0)
    unit.activate(0.0)
    unit.observe(observation(0.2), 1.0)

    assert unit.release_due(2.0) is False
    assert unit.release_due(5.0) is True


def test_maximum_priority_duration_forces_release():
    unit = machine(max_priority_s=10.0, min_green_s=2.0)
    unit.activate(0.0)
    unit.observe(observation(1.0), 9.0)

    assert unit.release_due(9.9) is False
    assert unit.release_due(10.0) is True


# ---------------------------------------------------------------------------
# Clearance sequence
# ---------------------------------------------------------------------------
def test_clearing_sequence_is_yellow_then_red_then_normal():
    unit = machine(yellow_s=1.0, all_red_s=1.0)
    unit.activate(0.0)
    unit.begin_clearing(0.0)

    assert unit.colour is SignalColor.YELLOW
    assert unit.tick(0.5) == []

    red_commands = unit.tick(1.0)
    assert len(red_commands) == 1
    assert red_commands[0].colour is SignalColor.RED
    assert unit.colour is SignalColor.RED

    assert unit.tick(1.5) == []

    normal_commands = unit.tick(2.0)
    assert len(normal_commands) == 1
    assert normal_commands[0].phase is PriorityPhase.NORMAL
    assert unit.phase is PriorityPhase.NORMAL
    assert unit.colour is SignalColor.RED


def test_clearing_skipped_step_still_ends_at_normal():
    unit = machine(yellow_s=1.0, all_red_s=1.0)
    unit.activate(0.0)
    unit.begin_clearing(0.0)

    commands = unit.tick(10.0)

    assert [command.colour for command in commands] == [SignalColor.RED, SignalColor.RED]
    assert [command.phase for command in commands] == [
        PriorityPhase.CLEARING,
        PriorityPhase.NORMAL,
    ]


def test_command_serialises():
    unit = machine()
    payload = unit.activate(5.0).to_dict()

    assert payload["colour"] == "green"
    assert payload["phase"] == "emergency_priority"
    assert payload["timestamp_iso"].endswith("+00:00")


# ---------------------------------------------------------------------------
# Fault handling
# ---------------------------------------------------------------------------
def test_fault_forces_red_and_blocks_observations():
    unit = machine()
    command = unit.force_fault(2.0, "model failure")

    assert command is not None
    assert unit.phase is PriorityPhase.FAULT
    assert unit.colour is SignalColor.RED
    assert unit.observe(observation(1.0), 3.0) is None
    assert unit.force_fault(4.0, "again") is None


def test_reset_recovers_to_normal_red():
    unit = machine()
    unit.force_fault(2.0, "sensor failure")

    command = unit.reset(3.0, "acknowledged")

    assert command is not None
    assert unit.phase is PriorityPhase.NORMAL
    assert unit.colour is SignalColor.RED
    assert unit.reset(4.0) is None


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------
def test_valid_transitions_pass():
    validate_transition(
        PriorityPhase.NORMAL, PriorityPhase.EMERGENCY_PRIORITY, SignalColor.RED, SignalColor.GREEN
    )
    validate_transition(
        PriorityPhase.EMERGENCY_PRIORITY, PriorityPhase.CLEARING, SignalColor.GREEN, SignalColor.YELLOW
    )
    validate_transition(
        PriorityPhase.CLEARING, PriorityPhase.CLEARING, SignalColor.YELLOW, SignalColor.RED
    )
    validate_transition(
        PriorityPhase.CLEARING, PriorityPhase.NORMAL, SignalColor.RED, SignalColor.RED
    )
    validate_transition(PriorityPhase.FAULT, PriorityPhase.NORMAL, SignalColor.RED, SignalColor.RED)


@pytest.mark.parametrize(
    "old_phase,new_phase,old_colour,new_colour",
    [
        (PriorityPhase.EMERGENCY_PRIORITY, PriorityPhase.NORMAL, SignalColor.GREEN, SignalColor.RED),
        (PriorityPhase.NORMAL, PriorityPhase.EMERGENCY_PRIORITY, SignalColor.RED, SignalColor.YELLOW),
        (PriorityPhase.CLEARING, PriorityPhase.NORMAL, SignalColor.YELLOW, SignalColor.RED),
        (PriorityPhase.CLEARING, PriorityPhase.EMERGENCY_PRIORITY, SignalColor.YELLOW, SignalColor.GREEN),
        (PriorityPhase.NORMAL, PriorityPhase.FAULT, SignalColor.RED, SignalColor.GREEN),
        (PriorityPhase.FAULT, PriorityPhase.NORMAL, SignalColor.RED, SignalColor.GREEN),
    ],
)
def test_invalid_transitions_raise(old_phase, new_phase, old_colour, new_colour):
    with pytest.raises(InvalidTransitionError):
        validate_transition(old_phase, new_phase, old_colour, new_colour)


def test_snapshot_reflects_state():
    unit = machine()
    unit.activate(1.0)

    snapshot = unit.snapshot()

    assert snapshot.approach == "Road 1"
    assert snapshot.colour is SignalColor.GREEN
    assert snapshot.phase is PriorityPhase.EMERGENCY_PRIORITY
    assert snapshot.priority_started_at == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Additional transition and dataclass validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "old_phase,new_phase,old_colour,new_colour",
    [
        (PriorityPhase.NORMAL, PriorityPhase.NORMAL, SignalColor.RED, SignalColor.GREEN),
        (PriorityPhase.NORMAL, PriorityPhase.EMERGENCY_PRIORITY, SignalColor.GREEN, SignalColor.GREEN),
        (PriorityPhase.EMERGENCY_PRIORITY, PriorityPhase.EMERGENCY_PRIORITY, SignalColor.GREEN, SignalColor.RED),
        (PriorityPhase.EMERGENCY_PRIORITY, PriorityPhase.CLEARING, SignalColor.RED, SignalColor.YELLOW),
        (PriorityPhase.CLEARING, PriorityPhase.CLEARING, SignalColor.RED, SignalColor.YELLOW),
        (PriorityPhase.CLEARING, PriorityPhase.CLEARING, SignalColor.GREEN, SignalColor.RED),
        (PriorityPhase.NORMAL, PriorityPhase.FAULT, SignalColor.RED, SignalColor.YELLOW),
        (PriorityPhase.FAULT, PriorityPhase.FAULT, SignalColor.RED, SignalColor.GREEN),
        (PriorityPhase.CLEARING, PriorityPhase.NORMAL, SignalColor.YELLOW, SignalColor.RED),
    ],
)
def test_additional_invalid_transitions_raise(old_phase, new_phase, old_colour, new_colour):
    with pytest.raises(InvalidTransitionError):
        validate_transition(old_phase, new_phase, old_colour, new_colour)


def test_same_phase_same_colour_is_a_valid_no_op():
    for phase, colour in (
        (PriorityPhase.NORMAL, SignalColor.RED),
        (PriorityPhase.EMERGENCY_PRIORITY, SignalColor.GREEN),
        (PriorityPhase.CLEARING, SignalColor.YELLOW),
        (PriorityPhase.CLEARING, SignalColor.RED),
        (PriorityPhase.FAULT, SignalColor.RED),
    ):
        validate_transition(phase, phase, colour, colour)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"approach": ""}, "approach"),
        ({"requested": 1}, "requested"),
        ({"confidence": "high"}, "confidence"),
        ({"confidence": float("nan")}, "confidence"),
        ({"confidence": 1.1}, "confidence"),
        ({"note": 5}, "note"),
    ],
)
def test_approach_observation_rejects_malformed_fields(kwargs, match):
    base = dict(approach="Road 1", requested=True, confidence=0.8, note="")
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ApproachObservation(**base)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"approach": " "}, "approach"),
        ({"colour": "red"}, "colour"),
        ({"phase": "normal"}, "phase"),
        ({"timestamp": -1.0}, "timestamp"),
        ({"timestamp": True}, "timestamp"),
        ({"reason": "  "}, "reason"),
        ({"command_id": "  "}, "command_id"),
    ],
)
def test_signal_command_rejects_malformed_fields(kwargs, match):
    base = dict(
        approach="Road 1",
        colour=SignalColor.RED,
        phase=PriorityPhase.NORMAL,
        timestamp=0.0,
        reason="initial",
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SignalCommand(**base)


def test_signal_command_and_snapshot_serialise():
    command = SignalCommand(
        "Road 1", SignalColor.GREEN, PriorityPhase.EMERGENCY_PRIORITY, 2.0, "confirmed"
    )
    payload = command.to_dict()
    assert payload["approach"] == "Road 1"
    assert payload["colour"] == "green"
    assert payload["phase"] == "emergency_priority"
    assert payload["timestamp"] == pytest.approx(2.0)
    assert payload["timestamp_iso"].endswith("+00:00")

    snapshot = ApproachSnapshot(
        "Road 1", SignalColor.GREEN, PriorityPhase.EMERGENCY_PRIORITY, 1.0, 1.0, "confirmed"
    )
    assert snapshot.to_dict()["priority_started_at"] == pytest.approx(1.0)
    assert snapshot.to_dict()["colour"] == "green"
