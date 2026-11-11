"""Per-approach priority machine.

The machine owns timing and hysteresis for one approach:

* temporal confirmation before a request becomes ready,
* activation threshold / release threshold hysteresis,
* release grace period,
* minimum green and maximum priority duration,
* the GREEN -> YELLOW -> RED clearance sequence.

Cross-approach arbitration is handled by `SignalController` in `arbiter.py`.
"""

from __future__ import annotations

import math
from typing import Optional

from ambulance_system.config import SignalConfig
from ambulance_system.logging_utils import get_logger
from ambulance_system.traffic_signal.states import (
    ApproachObservation,
    ApproachSnapshot,
    InvalidTransitionError,
    PriorityPhase,
    SignalColor,
    SignalCommand,
    validate_transition,
)


class ApproachMachine:
    """Timing and transition logic for a single traffic approach."""

    def __init__(
        self,
        approach: str,
        config: SignalConfig,
        now: float,
        *,
        logger=None,
    ) -> None:
        if not isinstance(approach, str) or not approach.strip():
            raise ValueError("approach must be a non-empty string")
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("now must be finite and >= 0")
        self.approach = approach
        self._config = config
        self._now = float(now)

        self._phase = PriorityPhase.NORMAL
        self._colour = SignalColor.RED
        self._priority_started_at: Optional[float] = None
        self._clearing_started_at: Optional[float] = None
        self._above_since: Optional[float] = None
        self._below_since: Optional[float] = None

        self.red_since = float(now)
        self.last_change_at = float(now)
        self.reason = "initial state"
        self._logger = logger or get_logger("traffic_signal")

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------
    @property
    def phase(self) -> PriorityPhase:
        return self._phase

    @property
    def colour(self) -> SignalColor:
        return self._colour

    @property
    def priority_started_at(self) -> Optional[float]:
        return self._priority_started_at

    def snapshot(self) -> ApproachSnapshot:
        return ApproachSnapshot(
            approach=self.approach,
            colour=self._colour,
            phase=self._phase,
            priority_started_at=self._priority_started_at,
            last_change_at=self.last_change_at,
            reason=self.reason,
        )

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------
    def observe(self, observation: ApproachObservation, now: float) -> Optional[float]:
        """Feed one observation.

        Returns the timestamp at which this approach's request became ready
        (confirmation complete), or `None` while it is not ready.
        """
        self._now = float(now)

        if self._phase is PriorityPhase.FAULT:
            return None

        if self._phase is PriorityPhase.EMERGENCY_PRIORITY:
            if not observation.requested or observation.confidence < self._config.release_threshold:
                if self._below_since is None:
                    self._below_since = self._now
            else:
                self._below_since = None
            return None

        requesting = (
            observation.requested and observation.confidence >= self._config.activation_threshold
        )
        if not requesting:
            self._above_since = None
            return None

        if self._above_since is None:
            self._above_since = self._now

        ready_at = self._above_since + self._config.min_confirmation_s
        if self._now >= ready_at:
            return ready_at
        return None

    def release_due(self, now: float) -> bool:
        """True when an active priority must start clearing."""
        if self._phase is not PriorityPhase.EMERGENCY_PRIORITY:
            return False
        if self._priority_started_at is None:
            return False

        now = float(now)
        if now - self._priority_started_at >= self._config.max_priority_s:
            return True

        if self._below_since is not None and now - self._below_since >= self._config.release_grace_s:
            if now - self._priority_started_at >= self._config.min_green_s:
                return True
        return False

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def activate(self, now: float, reason: str = "priority confirmed") -> SignalCommand:
        if self._phase is not PriorityPhase.NORMAL:
            raise InvalidTransitionError(
                f"cannot activate priority from {self._phase.value}"
            )
        command = self._emit(PriorityPhase.EMERGENCY_PRIORITY, SignalColor.GREEN, now, reason)
        self._priority_started_at = float(now)
        self._clearing_started_at = None
        self._above_since = None
        self._below_since = None
        return command

    def begin_clearing(self, now: float, reason: str = "release condition met") -> SignalCommand:
        if self._phase is not PriorityPhase.EMERGENCY_PRIORITY:
            raise InvalidTransitionError(
                f"cannot begin clearing from {self._phase.value}"
            )
        command = self._emit(PriorityPhase.CLEARING, SignalColor.YELLOW, now, reason)
        self._clearing_started_at = float(now)
        self._below_since = None
        return command

    def tick(self, now: float) -> list[SignalCommand]:
        """Advance the clearance sequence. Safe to call every decision cycle."""
        if self._phase is not PriorityPhase.CLEARING or self._clearing_started_at is None:
            return []

        now = float(now)
        elapsed = now - self._clearing_started_at
        if elapsed < 0.0:
            raise ValueError("clearing time cannot move backwards")

        commands: list[SignalCommand] = []

        if self._colour is SignalColor.YELLOW and elapsed >= self._config.yellow_s:
            commands.append(
                self._emit(
                    PriorityPhase.CLEARING,
                    SignalColor.RED,
                    now,
                    "yellow interval complete",
                    red_since=self._clearing_started_at + self._config.yellow_s,
                )
            )

        if (
            self._colour is SignalColor.RED
            and elapsed >= self._config.yellow_s + self._config.all_red_s
        ):
            commands.append(
                self._emit(
                    PriorityPhase.NORMAL,
                    SignalColor.RED,
                    now,
                    "clearance complete",
                    red_since=self._clearing_started_at + self._config.yellow_s,
                )
            )
            self._clearing_started_at = None
            self._priority_started_at = None

        return commands

    def force_fault(self, now: float, reason: str) -> Optional[SignalCommand]:
        """Force FAULT and all-red. Idempotent."""
        if self._phase is PriorityPhase.FAULT:
            return None
        command = self._emit(PriorityPhase.FAULT, SignalColor.RED, now, reason, red_since=now)
        self._clear_timing()
        return command

    def reset(self, now: float, reason: str = "manual reset") -> Optional[SignalCommand]:
        """Explicit emergency recovery to all-red NORMAL.

        This bypasses the normal transition table because every state can be
        safely exited to all-red. It is only reachable through the controller's
        `emergency_reset` / `clear_fault` methods.
        """
        self._clear_timing()
        if self._phase is PriorityPhase.NORMAL and self._colour is SignalColor.RED:
            return None
        return self._emit(
            PriorityPhase.NORMAL,
            SignalColor.RED,
            now,
            reason,
            red_since=now,
            validate=False,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _clear_timing(self) -> None:
        self._priority_started_at = None
        self._clearing_started_at = None
        self._above_since = None
        self._below_since = None

    def _emit(
        self,
        new_phase: PriorityPhase,
        new_colour: SignalColor,
        now: float,
        reason: str,
        *,
        red_since: Optional[float] = None,
        validate: bool = True,
    ) -> SignalCommand:
        now = float(now)
        if validate:
            validate_transition(self._phase, new_phase, self._colour, new_colour)

        old_colour = self._colour
        self._phase = new_phase
        self._colour = new_colour
        self.last_change_at = now
        self.reason = reason
        if red_since is not None:
            self.red_since = float(red_since)
        elif new_colour is SignalColor.RED and old_colour is not SignalColor.RED:
            self.red_since = now

        return SignalCommand(
            approach=self.approach,
            colour=new_colour,
            phase=new_phase,
            timestamp=now,
            reason=reason,
        )
