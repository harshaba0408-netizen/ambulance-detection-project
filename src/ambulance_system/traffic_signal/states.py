"""States, transitions and command objects for the simulated signal controller.

Two concepts are deliberately separated:

* `SignalColor` is what a driver sees (RED / YELLOW / GREEN).
* `PriorityPhase` is what the controller is doing (NORMAL / EMERGENCY_PRIORITY /
  CLEARING / FAULT).

Keeping them apart makes illegal combinations explicit: a CLEARING approach may
be YELLOW or RED, but never GREEN, and an emergency approach is always GREEN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from ambulance_system.events import new_event_id, utc_iso


class SignalColor(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class PriorityPhase(str, Enum):
    NORMAL = "normal"
    EMERGENCY_PRIORITY = "emergency_priority"
    CLEARING = "clearing"
    FAULT = "fault"


ALLOWED_PHASE_TRANSITIONS: Mapping[PriorityPhase, frozenset[PriorityPhase]] = {
    PriorityPhase.NORMAL: frozenset(
        {PriorityPhase.NORMAL, PriorityPhase.EMERGENCY_PRIORITY, PriorityPhase.FAULT}
    ),
    PriorityPhase.EMERGENCY_PRIORITY: frozenset(
        {PriorityPhase.EMERGENCY_PRIORITY, PriorityPhase.CLEARING, PriorityPhase.FAULT}
    ),
    PriorityPhase.CLEARING: frozenset(
        {PriorityPhase.CLEARING, PriorityPhase.NORMAL, PriorityPhase.FAULT}
    ),
    PriorityPhase.FAULT: frozenset({PriorityPhase.FAULT, PriorityPhase.NORMAL}),
}


class InvalidTransitionError(ValueError):
    """Raised when a state or colour change would violate the transition table."""


def _colour_allowed(phase: PriorityPhase, colour: SignalColor) -> bool:
    if phase is PriorityPhase.NORMAL:
        return colour is SignalColor.RED
    if phase is PriorityPhase.EMERGENCY_PRIORITY:
        return colour is SignalColor.GREEN
    if phase is PriorityPhase.CLEARING:
        return colour in (SignalColor.YELLOW, SignalColor.RED)
    return colour is SignalColor.RED


def validate_transition(
    old_phase: PriorityPhase,
    new_phase: PriorityPhase,
    old_colour: SignalColor,
    new_colour: SignalColor,
) -> None:
    """Validate one transition against the explicit matrix.

    Raises `InvalidTransitionError` with a readable message. This function is
    intentionally public so tests and future hardware adapters can reuse it.
    """
    if new_phase not in ALLOWED_PHASE_TRANSITIONS[old_phase]:
        raise InvalidTransitionError(
            f"phase transition {old_phase.value} -> {new_phase.value} is not allowed"
        )
    if not _colour_allowed(new_phase, new_colour):
        raise InvalidTransitionError(
            f"colour {new_colour.value} is not valid in phase {new_phase.value}"
        )

    if old_phase is new_phase:
        if old_phase is PriorityPhase.CLEARING:
            if (old_colour, new_colour) not in (
                (SignalColor.YELLOW, SignalColor.YELLOW),
                (SignalColor.YELLOW, SignalColor.RED),
                (SignalColor.RED, SignalColor.RED),
            ):
                raise InvalidTransitionError(
                    f"clearing cannot go {old_colour.value} -> {new_colour.value}"
                )
        elif old_colour is not new_colour:
            raise InvalidTransitionError(
                f"colour cannot change within phase {new_phase.value}"
            )
        return

    # Cross-phase colour rules.
    if old_phase is PriorityPhase.NORMAL and new_phase is PriorityPhase.EMERGENCY_PRIORITY:
        if old_colour is not SignalColor.RED or new_colour is not SignalColor.GREEN:
            raise InvalidTransitionError("priority can only be granted from RED to GREEN")
    if old_phase is PriorityPhase.EMERGENCY_PRIORITY and new_phase is PriorityPhase.CLEARING:
        if old_colour is not SignalColor.GREEN or new_colour is not SignalColor.YELLOW:
            raise InvalidTransitionError("priority must pass through GREEN -> YELLOW")
    if new_phase is PriorityPhase.FAULT and new_colour is not SignalColor.RED:
        raise InvalidTransitionError("FAULT must force RED")
    if new_phase is PriorityPhase.NORMAL and new_colour is not SignalColor.RED:
        raise InvalidTransitionError("NORMAL must be RED in this simulator")
    if (
        old_phase is PriorityPhase.CLEARING
        and new_phase is PriorityPhase.NORMAL
        and old_colour is not SignalColor.RED
    ):
        raise InvalidTransitionError("clearing must reach RED before returning to NORMAL")


@dataclass(frozen=True)
class ApproachObservation:
    """One decision cycle's evidence for one approach.

    `confidence` is the effective confidence supplied by the fusion layer in
    Stage 7. Until then, callers may pass the winning modality's confidence.
    """

    approach: str
    requested: bool
    confidence: float
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.approach, str) or not self.approach.strip():
            raise ValueError("approach must be a non-empty string")
        if not isinstance(self.requested, bool):
            raise ValueError("requested must be a bool")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be a finite number within [0.0, 1.0]")
        if not isinstance(self.note, str):
            raise ValueError("note must be a string")


@dataclass(frozen=True)
class SignalCommand:
    """An emitted, traceable output change for one approach."""

    approach: str
    colour: SignalColor
    phase: PriorityPhase
    timestamp: float
    reason: str
    command_id: str = field(default_factory=new_event_id)

    def __post_init__(self) -> None:
        if not isinstance(self.approach, str) or not self.approach.strip():
            raise ValueError("approach must be a non-empty string")
        if not isinstance(self.colour, SignalColor):
            raise ValueError("colour must be a SignalColor")
        if not isinstance(self.phase, PriorityPhase):
            raise ValueError("phase must be a PriorityPhase")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
            or float(self.timestamp) < 0.0
        ):
            raise ValueError("timestamp must be finite and >= 0")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not isinstance(self.command_id, str) or not self.command_id.strip():
            raise ValueError("command_id must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "timestamp": float(self.timestamp),
            "timestamp_iso": utc_iso(float(self.timestamp)),
            "approach": self.approach,
            "colour": self.colour.value,
            "phase": self.phase.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ApproachSnapshot:
    """Immutable view of one approach for the dashboard and tests."""

    approach: str
    colour: SignalColor
    phase: PriorityPhase
    priority_started_at: Optional[float]
    last_change_at: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": self.approach,
            "colour": self.colour.value,
            "phase": self.phase.value,
            "priority_started_at": self.priority_started_at,
            "last_change_at": float(self.last_change_at),
            "reason": self.reason,
        }
