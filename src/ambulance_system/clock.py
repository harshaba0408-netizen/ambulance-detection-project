"""Injectable clock abstraction.

State machines must never call `time.sleep()` or `time.time()` directly if we
want deterministic tests. Components accept a `Clock`; production wires
`RealClock`, tests wire `FakeClock`.
"""

from __future__ import annotations

import math
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Minimal clock interface used by the pipeline and state machines."""

    def now(self) -> float:
        """Epoch seconds, comparable with log timestamps."""

    def monotonic(self) -> float:
        """Monotonic seconds, safe for measuring durations."""


class RealClock:
    """Wall-clock and monotonic time from the standard library."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()


class FakeClock:
    """Deterministic clock for tests and simulations."""

    def __init__(self, start: float = 0.0, *, monotonic_start: float | None = None) -> None:
        if not math.isfinite(start) or start < 0.0:
            raise ValueError("start must be finite and >= 0")
        self._now = float(start)
        self._monotonic = float(start if monotonic_start is None else monotonic_start)

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> float:
        """Move both clocks forward by `seconds` and return the new time."""
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ValueError("seconds must be a number")
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("seconds must be finite and >= 0")
        self._now += float(seconds)
        self._monotonic += float(seconds)
        return self._now
