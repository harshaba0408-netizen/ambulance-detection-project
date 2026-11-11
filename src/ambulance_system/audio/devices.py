"""Audio sources.

Only the offline file source is implemented in Stage 6. Live microphone capture
is deferred until `sounddevice` is approved; it must implement the same
`AudioSource` protocol when it is added.

`FileAudioSource` decodes the whole file once (the published clips are about
3 seconds) and then yields fixed-size blocks. That keeps the windowing code
identical for a future live source without pretending the file path is a
low-latency device.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ambulance_system.audio.features import load_audio


class AudioSourceError(RuntimeError):
    """Raised when an audio source cannot be opened or read."""


@runtime_checkable
class AudioSource(Protocol):
    """Minimal mono block source interface."""

    @property
    def name(self) -> str:
        ...

    @property
    def sample_rate(self) -> int:
        ...

    @property
    def is_open(self) -> bool:
        ...

    @property
    def exhausted(self) -> bool:
        ...

    def open(self) -> None:
        ...

    def read(self) -> Optional[np.ndarray]:
        ...

    def close(self) -> None:
        ...


class FileAudioSource:
    """Yield a recorded file as fixed-size mono blocks."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = 22050,
        block_s: float = 0.25,
        duration: Optional[float] = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"audio file not found: {self.path}")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive int")
        if (
            isinstance(block_s, bool)
            or not isinstance(block_s, (int, float))
            or not math.isfinite(float(block_s))
            or float(block_s) <= 0
        ):
            raise ValueError("block_s must be > 0")
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError("duration must be > 0 or None")

        self._sample_rate = sample_rate
        self._block_samples = max(1, int(round(float(block_s) * sample_rate)))
        self._duration = duration
        self._samples: Optional[np.ndarray] = None
        self._offset = 0
        self._exhausted = False

    @property
    def name(self) -> str:
        return f"file:{self.path.name}"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def block_samples(self) -> int:
        return self._block_samples

    @property
    def is_open(self) -> bool:
        return self._samples is not None

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def open(self) -> None:
        if self._samples is not None:
            return
        try:
            samples, rate = load_audio(
                self.path,
                sample_rate=self._sample_rate,
                duration=self._duration,
            )
        except Exception as exc:
            raise AudioSourceError(f"could not load audio source {self.name}: {exc}") from exc
        if rate != self._sample_rate:
            raise AudioSourceError(
                f"expected {self._sample_rate} Hz from {self.name}, got {rate} Hz"
            )
        self._samples = samples
        self._offset = 0
        self._exhausted = False

    def read(self) -> Optional[np.ndarray]:
        if self._samples is None:
            return None
        if self._offset >= self._samples.size:
            self._exhausted = True
            return None
        end = min(self._offset + self._block_samples, self._samples.size)
        block = self._samples[self._offset : end]
        self._offset = end
        return block

    def close(self) -> None:
        self._samples = None
        self._offset = 0
