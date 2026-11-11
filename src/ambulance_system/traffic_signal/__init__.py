"""Simulated traffic signal state machine (Stage 3).

Public API:

* `SignalController` — arbitration and safety invariants across approaches.
* `ApproachMachine` — timing and hysteresis for one approach.
* `PriorityPhase`, `SignalColor`, `SignalCommand` — state vocabulary.
"""

from ambulance_system.traffic_signal.arbiter import SafetyViolationError, SignalController
from ambulance_system.traffic_signal.machine import ApproachMachine
from ambulance_system.traffic_signal.states import (
    ALLOWED_PHASE_TRANSITIONS,
    ApproachObservation,
    ApproachSnapshot,
    InvalidTransitionError,
    PriorityPhase,
    SignalColor,
    SignalCommand,
    validate_transition,
)

__all__ = [
    "ALLOWED_PHASE_TRANSITIONS",
    "ApproachMachine",
    "ApproachObservation",
    "ApproachSnapshot",
    "InvalidTransitionError",
    "PriorityPhase",
    "SafetyViolationError",
    "SignalColor",
    "SignalCommand",
    "SignalController",
    "validate_transition",
]
