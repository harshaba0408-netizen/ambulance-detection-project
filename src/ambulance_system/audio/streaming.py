"""Sliding-window audio streaming over an `AudioSource`.

The legacy dashboard classified one three-second slice per file. This module
keeps the same MFCC/classifier pipeline but slides a window across the whole
file, so a siren that starts late is still observed. A future microphone source
feeds the same windowing and classifier code; no microphone capture is included
in Stage 6.

Windows are emitted at `hop = window_s - overlap_s`. The final window may be
shorter than `window_s`; it is marked `partial=True` and is classified using the
samples that actually exist (no zero-padding, which would change the features
relative to the legacy one-shot path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np

from ambulance_system.audio.classifier import SirenClassifier
from ambulance_system.audio.devices import AudioSource, FileAudioSource
from ambulance_system.config import AudioConfig
from ambulance_system.events import DetectionEvent


@dataclass(frozen=True)
class AudioWindow:
    """A slice of mono audio with its position in the stream."""

    index: int
    start_s: float
    end_s: float
    samples: np.ndarray
    sample_rate: int
    partial: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative int")
        if (
            isinstance(self.start_s, bool)
            or not isinstance(self.start_s, (int, float))
            or not np.isfinite(float(self.start_s))
            or float(self.start_s) < 0.0
        ):
            raise ValueError("start_s must be finite and >= 0")
        if (
            isinstance(self.end_s, bool)
            or not isinstance(self.end_s, (int, float))
            or not np.isfinite(float(self.end_s))
            or float(self.end_s) <= float(self.start_s)
        ):
            raise ValueError("end_s must be finite and greater than start_s")
        if not isinstance(self.samples, np.ndarray):
            raise ValueError("samples must be a numpy array")
        if self.samples.ndim != 1 or self.samples.size == 0:
            raise ValueError("samples must be a non-empty 1-D array")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("samples must contain only finite values")
        if not isinstance(self.sample_rate, int) or isinstance(self.sample_rate, bool) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive int")
        if not isinstance(self.partial, bool):
            raise ValueError("partial must be a bool")

    @property
    def duration_s(self) -> float:
        return float(self.end_s) - float(self.start_s)


@dataclass(frozen=True)
class WindowPrediction:
    """Classifier output for one `AudioWindow`."""

    window: AudioWindow
    is_siren: bool
    confidence: Optional[float]

    def __post_init__(self) -> None:
        if not isinstance(self.window, AudioWindow):
            raise ValueError("window must be an AudioWindow")
        if not isinstance(self.is_siren, bool):
            raise ValueError("is_siren must be a bool")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not np.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be within [0.0, 1.0] or None")

    def to_event(self, approach: str) -> DetectionEvent:
        """Convert to the shared audio event schema.

        When the estimator has no `predict_proba`, the hard label is used as a
        0.0/1.0 confidence and that limitation is visible in the value.
        """
        confidence = float(self.is_siren) if self.confidence is None else float(self.confidence)
        return DetectionEvent.audio(
            approach=approach,
            confidence=confidence,
            window_start_s=float(self.window.start_s),
            window_end_s=float(self.window.end_s),
        )


@dataclass(frozen=True)
class StreamSummary:
    """Aggregate counts for one stream, used for logging and tests."""

    windows: int
    siren_windows: int
    max_confidence: Optional[float]

    def __post_init__(self) -> None:
        if self.windows < 0 or self.siren_windows < 0:
            raise ValueError("window counts must be >= 0")
        if self.siren_windows > self.windows:
            raise ValueError("siren_windows cannot exceed windows")
        if self.max_confidence is not None and not 0.0 <= float(self.max_confidence) <= 1.0:
            raise ValueError("max_confidence must be within [0.0, 1.0] or None")

    @property
    def detected(self) -> bool:
        return self.siren_windows > 0


def _window_geometry(sample_rate: int, window_s: float, overlap_s: float) -> tuple[int, int]:
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive int")
    if (
        isinstance(window_s, bool)
        or not isinstance(window_s, (int, float))
        or not np.isfinite(float(window_s))
        or float(window_s) <= 0
    ):
        raise ValueError("window_s must be > 0")
    if (
        isinstance(overlap_s, bool)
        or not isinstance(overlap_s, (int, float))
        or not np.isfinite(float(overlap_s))
        or float(overlap_s) < 0
    ):
        raise ValueError("overlap_s must be >= 0")
    if float(overlap_s) >= float(window_s):
        raise ValueError("overlap_s must be smaller than window_s")

    window_samples = max(1, int(round(float(window_s) * sample_rate)))
    hop_samples = max(1, int(round((float(window_s) - float(overlap_s)) * sample_rate)))
    return window_samples, hop_samples


def iter_windows(
    samples: np.ndarray,
    *,
    sample_rate: int,
    window_s: float,
    overlap_s: float,
    include_partial: bool = True,
) -> Iterator[AudioWindow]:
    """Slide a window across an in-memory mono signal."""
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a numpy array")
    if samples.ndim != 1:
        raise ValueError("samples must be a 1-D mono array")
    if samples.size == 0:
        raise ValueError("samples must not be empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values")

    window_samples, hop_samples = _window_geometry(sample_rate, window_s, overlap_s)
    total = samples.size
    index = 0
    start = 0
    while start < total:
        end = min(start + window_samples, total)
        partial = end - start < window_samples
        if partial and not include_partial:
            break
        yield AudioWindow(
            index=index,
            start_s=start / sample_rate,
            end_s=end / sample_rate,
            samples=samples[start:end].copy(),
            sample_rate=sample_rate,
            partial=partial,
        )
        index += 1
        start += hop_samples


def stream_predictions(
    source: AudioSource,
    classifier: SirenClassifier,
    *,
    window_s: float,
    overlap_s: float,
    include_partial: bool = True,
) -> Iterator[WindowPrediction]:
    """Read blocks from `source`, classifying each sliding window.

    The source is opened if necessary and always closed by this generator.
    """
    if not isinstance(classifier, SirenClassifier):
        raise TypeError("classifier must be a SirenClassifier")
    window_samples, hop_samples = _window_geometry(
        source.sample_rate, window_s, overlap_s
    )

    opened_here = not source.is_open
    if opened_here:
        source.open()
    try:
        buffer = np.empty(0, dtype=np.float32)
        index = 0
        while True:
            block = source.read()
            if block is None:
                break
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 1:
                raise ValueError("audio blocks must be 1-D")
            if block.size == 0:
                continue
            if not np.all(np.isfinite(block)):
                raise ValueError("audio blocks must contain only finite values")
            if buffer.size:
                buffer = np.concatenate((buffer, block))
            else:
                buffer = block
            while buffer.size >= window_samples:
                window = AudioWindow(
                    index=index,
                    start_s=index * hop_samples / source.sample_rate,
                    end_s=(index * hop_samples + window_samples) / source.sample_rate,
                    samples=buffer[:window_samples].copy(),
                    sample_rate=source.sample_rate,
                )
                yield _classify(window, classifier)
                index += 1
                buffer = buffer[hop_samples:]

        if include_partial and buffer.size:
            start_samples = index * hop_samples
            end_samples = start_samples + buffer.size
            window = AudioWindow(
                index=index,
                start_s=start_samples / source.sample_rate,
                end_s=end_samples / source.sample_rate,
                samples=buffer.copy(),
                sample_rate=source.sample_rate,
                partial=True,
            )
            yield _classify(window, classifier)
    finally:
        if opened_here:
            source.close()


def _classify(window: AudioWindow, classifier: SirenClassifier) -> WindowPrediction:
    prediction = classifier.predict_samples(window.samples)
    return WindowPrediction(
        window=window,
        is_siren=prediction.is_siren,
        confidence=prediction.confidence,
    )


def analyze_file(
    path: str,
    classifier: SirenClassifier,
    config: AudioConfig,
    *,
    include_partial: bool = True,
) -> list[WindowPrediction]:
    """Classify every sliding window in one file."""
    if classifier.sample_rate != config.sample_rate:
        raise ValueError(
            "classifier.sample_rate must match audio.sample_rate "
            f"({classifier.sample_rate} != {config.sample_rate})"
        )
    source = FileAudioSource(
        path,
        sample_rate=config.sample_rate,
        block_s=config.block_s,
    )
    try:
        return list(
            stream_predictions(
                source,
                classifier,
                window_s=config.window_s,
                overlap_s=config.overlap_s,
                include_partial=include_partial,
            )
        )
    finally:
        source.close()


def summarize(predictions: Iterable[WindowPrediction]) -> StreamSummary:
    """Aggregate window predictions into counts and a maximum confidence."""
    items = list(predictions)
    confidences = [
        float(item.confidence) for item in items if item.confidence is not None
    ]
    return StreamSummary(
        windows=len(items),
        siren_windows=sum(1 for item in items if item.is_siren),
        max_confidence=max(confidences) if confidences else None,
    )
