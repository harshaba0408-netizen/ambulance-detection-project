"""Measured rates, latency percentiles and resource sampling.

All values come from the running process. Nothing here estimates performance
from model metadata.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from ambulance_system.clock import Clock, RealClock


class RateMeter:
    """Counts events and reports a measured rate since the last reset."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock = clock or RealClock()
        self._start = self._clock.monotonic()
        self._count = 0

    def reset(self, start: Optional[float] = None) -> None:
        self._start = self._clock.monotonic() if start is None else float(start)
        self._count = 0

    def mark(self, count: int = 1) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative int")
        self._count += count

    @property
    def count(self) -> int:
        return self._count

    def rate(self) -> float:
        elapsed = self._clock.monotonic() - self._start
        if elapsed <= 0:
            return 0.0
        return self._count / elapsed


class LatencyTracker:
    """Rolling window of latencies in milliseconds."""

    def __init__(self, max_samples: int = 256) -> None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("max_samples must be an int >= 1")
        self._samples: deque[float] = deque(maxlen=max_samples)

    def reset(self) -> None:
        self._samples.clear()

    def add(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ValueError("latency must be a number")
        if not math.isfinite(float(seconds)) or float(seconds) < 0:
            raise ValueError("latency must be finite and >= 0")
        self._samples.append(float(seconds) * 1000.0)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def average_ms(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def p95_ms(self) -> float:
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]


@dataclass(frozen=True)
class ResourceSnapshot:
    """One sample of system and process resource usage."""

    timestamp: float
    cpu_percent: Optional[float]
    system_ram_percent: Optional[float]
    system_ram_used_mb: Optional[float]
    process_ram_mb: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "system_ram_percent": self.system_ram_percent,
            "system_ram_used_mb": self.system_ram_used_mb,
            "process_ram_mb": self.process_ram_mb,
        }


class ResourceMonitor:
    """Samples CPU and memory through psutil when available."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock = clock or RealClock()
        try:
            import psutil  # noqa: F401

            self._psutil = psutil
            self._process = psutil.Process()
        except Exception:
            self._psutil = None
            self._process = None

    def sample(self) -> ResourceSnapshot:
        timestamp = self._clock.now()
        if self._psutil is None:
            return ResourceSnapshot(timestamp, None, None, None, None)

        memory = self._psutil.virtual_memory()
        process_memory = self._process.memory_info().rss if self._process is not None else None
        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=float(self._psutil.cpu_percent(interval=None)),
            system_ram_percent=float(memory.percent),
            system_ram_used_mb=float(memory.used / 1024**2),
            process_ram_mb=float(process_memory / 1024**2) if process_memory is not None else None,
        )
