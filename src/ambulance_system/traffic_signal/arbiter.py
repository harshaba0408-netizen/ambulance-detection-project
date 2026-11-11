"""Cross-approach arbitration and safety invariants.

The controller is the only component allowed to grant GREEN. It enforces:

* at most one GREEN approach,
* no GREEN while any other approach is YELLOW,
* a full all-red interval between a green ending and the next green,
* deterministic conflict resolution (earliest confirmed request, then
  configured approach order),
* all-red FAULT state with explicit recovery.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from ambulance_system.clock import Clock, RealClock
from ambulance_system.config import SignalConfig
from ambulance_system.logging_utils import get_logger
from ambulance_system.traffic_signal.machine import ApproachMachine
from ambulance_system.traffic_signal.states import (
    ApproachObservation,
    ApproachSnapshot,
    PriorityPhase,
    SignalColor,
    SignalCommand,
)

_EPSILON = 1e-9


class SafetyViolationError(RuntimeError):
    """Raised if an invariant would be broken. Indicates a controller bug."""


class SignalController:
    """Deterministic four-approach signal controller for the simulator."""

    def __init__(
        self,
        config: SignalConfig,
        approaches: Sequence[str],
        *,
        clock: Optional[Clock] = None,
        logger=None,
    ) -> None:
        names = tuple(approaches)
        if not names:
            raise ValueError("at least one approach is required")
        if len(set(names)) != len(names):
            raise ValueError("approach names must be unique")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("approach names must be non-empty strings")

        self._config = config
        self._clock = clock or RealClock()
        self._logger = logger or get_logger("traffic_signal")

        now = float(self._clock.now())
        self._last_now = now
        self._machines: dict[str, ApproachMachine] = {
            name: ApproachMachine(name, config, now, logger=self._logger) for name in names
        }
        self._order = {name: index for index, name in enumerate(names)}
        self._pending: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------
    @property
    def states(self) -> dict[str, ApproachSnapshot]:
        return {name: machine.snapshot() for name, machine in self._machines.items()}

    @property
    def active_approach(self) -> Optional[str]:
        for name, machine in self._machines.items():
            if machine.colour is SignalColor.GREEN:
                return name
        return None

    @property
    def is_fault(self) -> bool:
        return any(machine.phase is PriorityPhase.FAULT for machine in self._machines.values())

    @property
    def pending_approaches(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in sorted(self._pending, key=lambda item: (self._pending[item], self._order[item]))
        )

    # ------------------------------------------------------------------
    # Decision cycle
    # ------------------------------------------------------------------
    def step(
        self,
        observations: Optional[Mapping[str, ApproachObservation]] = None,
        now: Optional[float] = None,
    ) -> list[SignalCommand]:
        """Run one decision cycle and return only the changes that occurred."""
        now = self._resolve_now(now)
        observations = self._validate_observations(observations)

        if self.is_fault:
            self._last_now = now
            return []

        commands: list[SignalCommand] = []

        # 1. Advance any clearance already in progress.
        for machine in self._machines.values():
            commands.extend(machine.tick(now))

        # 2. Feed evidence to every approach. A missing observation counts as
        #    no evidence, so a stale producer cannot hold priority forever.
        ready: dict[str, float] = {}
        for name, machine in self._machines.items():
            observation = observations.get(
                name, ApproachObservation(name, False, 0.0, "no observation")
            )
            ready_at = machine.observe(observation, now)
            if ready_at is not None:
                ready[name] = ready_at

        # 3. Release an active priority if its conditions have lapsed.
        for name, machine in self._machines.items():
            if machine.phase is PriorityPhase.EMERGENCY_PRIORITY and machine.release_due(now):
                reason = self._release_reason(machine, now)
                commands.append(machine.begin_clearing(now, reason))
                self._pending.pop(name, None)

        # 4. Keep only requests from approaches that are currently eligible.
        for name, machine in self._machines.items():
            ready_at = ready.get(name)
            if ready_at is not None and machine.phase is PriorityPhase.NORMAL:
                existing = self._pending.get(name)
                self._pending[name] = ready_at if existing is None else min(existing, ready_at)
            else:
                self._pending.pop(name, None)

        # 5. Grant at most one priority, and only after the all-red interval.
        if self._pending and self._safe_to_grant(now):
            name = min(self._pending, key=lambda item: (self._pending[item], self._order[item]))
            commands.append(self._machines[name].activate(now, "priority confirmed"))
            self._pending.pop(name, None)

        self._assert_safety()
        self._log_commands(commands)
        self._last_now = now
        return commands

    # ------------------------------------------------------------------
    # Explicit safety actions
    # ------------------------------------------------------------------
    def emergency_reset(self, reason: str = "manual emergency reset", now: Optional[float] = None) -> list[SignalCommand]:
        """Force every approach to all-red NORMAL and clear all timers."""
        now = self._resolve_now(now)
        commands = [
            command
            for command in (machine.reset(now, reason) for machine in self._machines.values())
            if command is not None
        ]
        self._pending.clear()
        self._assert_safety()
        self._logger.warning("emergency reset", extra={"event": {"reason": reason}})
        self._log_commands(commands)
        self._last_now = now
        return commands

    def clear_all(self, now: Optional[float] = None) -> list[SignalCommand]:
        """Alias kept for the dashboard's existing Clear All action."""
        return self.emergency_reset("manual Clear All", now=now)

    def trigger_fault(self, reason: str, now: Optional[float] = None) -> list[SignalCommand]:
        """Force all approaches to FAULT / all-red."""
        now = self._resolve_now(now)
        commands = [
            command
            for command in (machine.force_fault(now, reason) for machine in self._machines.values())
            if command is not None
        ]
        self._pending.clear()
        self._assert_safety()
        self._logger.error("SAFETY fault", extra={"event": {"reason": reason}})
        self._log_commands(commands)
        self._last_now = now
        return commands

    def clear_fault(self, reason: str = "fault acknowledged", now: Optional[float] = None) -> list[SignalCommand]:
        """Explicit recovery from FAULT. No-op when not in fault."""
        if not self.is_fault:
            return []
        return self.emergency_reset(reason, now=now)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_now(self, now: Optional[float]) -> float:
        value = float(self._clock.now() if now is None else now)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("now must be finite and >= 0")
        if value < self._last_now:
            raise ValueError(
                f"time cannot move backwards (last={self._last_now:.6f}, now={value:.6f})"
            )
        return value

    def _validate_observations(
        self, observations: Optional[Mapping[str, ApproachObservation]]
    ) -> Mapping[str, ApproachObservation]:
        if observations is None:
            return {}
        if not isinstance(observations, Mapping):
            raise TypeError("observations must be a mapping of approach -> ApproachObservation")
        unknown = set(observations) - set(self._machines)
        if unknown:
            raise ValueError(f"unknown approach(es): {', '.join(sorted(unknown))}")
        for name, observation in observations.items():
            if not isinstance(observation, ApproachObservation):
                raise TypeError(f"observation for {name!r} must be an ApproachObservation")
            if observation.approach != name:
                raise ValueError(
                    f"observation key {name!r} does not match approach {observation.approach!r}"
                )
        return observations

    def _safe_to_grant(self, now: float) -> bool:
        for machine in self._machines.values():
            if machine.phase is not PriorityPhase.NORMAL or machine.colour is not SignalColor.RED:
                return False
        newest_red = max(machine.red_since for machine in self._machines.values())
        return (now - newest_red) + _EPSILON >= self._config.all_red_s

    def _release_reason(self, machine: ApproachMachine, now: float) -> str:
        started = machine.priority_started_at
        if started is not None and now - started + _EPSILON >= self._config.max_priority_s:
            return "maximum priority duration reached"
        return "release condition met"

    def _assert_safety(self) -> None:
        greens = [name for name, machine in self._machines.items() if machine.colour is SignalColor.GREEN]
        yellows = [name for name, machine in self._machines.items() if machine.colour is SignalColor.YELLOW]
        if len(greens) > 1:
            raise SafetyViolationError(f"multiple green approaches: {greens}")
        if greens and yellows:
            raise SafetyViolationError(f"green {greens} simultaneous with yellow {yellows}")
        if self.is_fault and any(
            machine.colour is not SignalColor.RED for machine in self._machines.values()
        ):
            raise SafetyViolationError("FAULT must be all-red")

    def _log_commands(self, commands: list[SignalCommand]) -> None:
        for command in commands:
            self._logger.info(
                "signal command",
                extra={"event": command.to_dict(), "approach": command.approach},
            )
