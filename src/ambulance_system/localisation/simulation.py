"""Synthetic microphone-array signals with known propagation delays.

These helpers exist so GCC-PHAT and the bearing solver can be tested against a
ground truth without a real microphone array. The model is deliberately narrow:

* free field, no reverberation, no walls, no sensor directivity;
* a plane wave (far-field source) with a single propagation speed;
* independent white sensor noise;
* fractional delays applied as an FFT phase shift, which is a circular shift
  and therefore intended for analysis windows, not acoustic room simulation.

Nothing here loads a recording, opens an audio device or touches the priority
decision path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from ambulance_system.localisation.doa import SPEED_OF_SOUND_M_S


def _validate_signal(signal: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional signal with at least two samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def _validate_positions(
    positions_m: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    if not positions_m:
        raise ValueError("at least one sensor position is required")
    positions: dict[str, np.ndarray] = {}
    for name, position in positions_m.items():
        label = str(name).strip()
        if not label:
            raise ValueError("sensor names must be non-empty")
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (2,):
            raise ValueError(f"sensor {name!r} must have exactly two coordinates")
        if not np.all(np.isfinite(point)):
            raise ValueError(f"sensor {name!r} has non-finite coordinates")
        positions[label] = point
    return positions


def _validate_rate(sample_rate: int) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError("sample_rate must be a positive integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return int(sample_rate)


def _validate_duration(duration_s: float) -> None:
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be positive and finite")


def plane_wave_arrival_times(
    positions_m: Mapping[str, Sequence[float]],
    bearing_deg: float,
    *,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> dict[str, float]:
    """Arrival time at each sensor for a far-field source.

    The bearing is measured from broadside (``+y``) towards ``+x``. Times are
    relative to the coordinate origin and may be negative.
    """
    if not np.isfinite(bearing_deg):
        raise ValueError("bearing_deg must be finite")
    if not np.isfinite(speed_of_sound_m_s) or speed_of_sound_m_s <= 0.0:
        raise ValueError("speed_of_sound_m_s must be positive and finite")
    positions = _validate_positions(positions_m)
    theta = np.radians(float(bearing_deg))
    unit_vector = np.array([np.sin(theta), np.cos(theta)], dtype=np.float64)
    return {
        name: float(-(point @ unit_vector) / float(speed_of_sound_m_s))
        for name, point in positions.items()
    }


def fractional_delay(
    signal: Sequence[float],
    delay_s: float,
    sample_rate: int,
) -> np.ndarray:
    """Delay a signal by a fractional number of samples via FFT phase rotation.

    A positive ``delay_s`` makes the output later. The operation is circular:
    samples shifted past the end wrap to the start. Keep the delay much shorter
    than the window when that matters.
    """
    values = _validate_signal(signal, "signal")
    sample_rate = _validate_rate(sample_rate)
    if not np.isfinite(delay_s):
        raise ValueError("delay_s must be finite")
    if delay_s == 0.0:
        return values.copy()

    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sample_rate)
    shifted = spectrum * np.exp(-2j * np.pi * frequencies * float(delay_s))
    return np.fft.irfft(shifted, n=values.size)


def _unit_rms_source(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    source = rng.standard_normal(length)
    rms = float(np.sqrt(np.mean(source**2)))
    if rms <= 0.0:
        raise ValueError("generated source has zero variance")
    return source / rms


def _noise(
    length: int,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    if noise_std < 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be non-negative and finite")
    if noise_std == 0.0:
        return np.zeros(length, dtype=np.float64)
    return rng.normal(0.0, float(noise_std), size=length)


@dataclass(frozen=True)
class ArraySignals:
    """Synthetic channels for one plane-wave source plus sensor noise."""

    signals: Mapping[str, np.ndarray]
    sample_rate: int
    speed_of_sound_m_s: float
    bearing_deg: float
    arrival_times_s: Mapping[str, float]

    def delay_between(self, reference: str, moving: str) -> float:
        """Ground-truth delay of ``moving`` relative to ``reference``."""
        return float(
            self.arrival_times_s[moving] - self.arrival_times_s[reference]
        )


@dataclass(frozen=True)
class TwoSourceSignals:
    """Synthetic channels for two independent plane-wave sources."""

    signals: Mapping[str, np.ndarray]
    sample_rate: int
    speed_of_sound_m_s: float
    bearings_deg: Tuple[float, ...]
    amplitudes: Tuple[float, ...]
    arrival_times_s: Tuple[Mapping[str, float], ...]

    def delay_between(self, reference: str, moving: str, source: int) -> float:
        """Ground-truth delay of ``moving`` relative to ``reference`` for one source."""
        times = self.arrival_times_s[source]
        return float(times[moving] - times[reference])


def simulate_plane_wave(
    positions_m: Mapping[str, Sequence[float]],
    bearing_deg: float,
    *,
    duration_s: float = 0.25,
    sample_rate: int = 16000,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
    noise_std: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    source: Optional[Sequence[float]] = None,
) -> ArraySignals:
    """Simulate one far-field source at a known bearing.

    Args:
        positions_m: Mapping of sensor name to ``(x, y)`` position in metres.
        bearing_deg: Source bearing from broadside towards ``+x``.
        duration_s: Signal length in seconds.
        sample_rate: Sample rate in Hz.
        speed_of_sound_m_s: Assumed propagation speed.
        noise_std: Standard deviation of independent Gaussian noise added to
            each channel. The source has unit RMS before propagation.
        rng: NumPy generator; pass a seeded generator for reproducibility.
        source: Optional source samples. Must contain at least
            ``round(duration_s * sample_rate)`` samples; the first ``n`` are
            used and normalised to unit RMS.

    Returns:
        :class:`ArraySignals` with one channel per sensor and the ground-truth
        arrival times.
    """
    _validate_duration(duration_s)
    sample_rate = _validate_rate(sample_rate)
    positions = _validate_positions(positions_m)
    arrival_times = plane_wave_arrival_times(
        positions,
        bearing_deg,
        speed_of_sound_m_s=speed_of_sound_m_s,
    )

    length = int(round(float(duration_s) * sample_rate))
    if length < 2:
        raise ValueError("duration_s and sample_rate must produce at least two samples")
    generator = rng if rng is not None else np.random.default_rng()

    if source is None:
        signal = _unit_rms_source(length, generator)
    else:
        provided = _validate_signal(source, "source")
        if provided.size < length:
            raise ValueError("source is shorter than the requested duration")
        signal = provided[:length]
        rms = float(np.sqrt(np.mean(signal**2)))
        if rms <= 0.0:
            raise ValueError("source has zero variance")
        signal = signal / rms

    channels: dict[str, np.ndarray] = {}
    for name, arrival in arrival_times.items():
        propagated = fractional_delay(signal, arrival, sample_rate)
        channels[name] = propagated + _noise(length, generator, noise_std)

    return ArraySignals(
        signals=channels,
        sample_rate=sample_rate,
        speed_of_sound_m_s=float(speed_of_sound_m_s),
        bearing_deg=float(bearing_deg),
        arrival_times_s=arrival_times,
    )


def simulate_two_sources(
    positions_m: Mapping[str, Sequence[float]],
    bearings_deg: Sequence[float],
    *,
    amplitudes: Optional[Sequence[float]] = None,
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
    noise_std: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> TwoSourceSignals:
    """Mix two independent far-field sources with known bearings.

    The two sources have independent white spectra, so their cross terms average
    down and GCC-PHAT can expose two delay peaks when the bearings are far enough
    apart. Equal amplitudes are not required.

    Args:
        positions_m: Sensor name to ``(x, y)`` position.
        bearings_deg: Two source bearings in degrees.
        amplitudes: Optional source amplitudes (default ``1.0`` each).
        duration_s: Signal length in seconds.
        sample_rate: Sample rate in Hz.
        speed_of_sound_m_s: Assumed propagation speed.
        noise_std: Per-channel Gaussian sensor noise standard deviation.
        rng: NumPy generator for reproducibility.

    Returns:
        :class:`TwoSourceSignals` containing the mixed channels and per-source
        ground-truth arrival times.
    """
    _validate_duration(duration_s)
    sample_rate = _validate_rate(sample_rate)
    positions = _validate_positions(positions_m)

    bearings = tuple(float(value) for value in bearings_deg)
    if len(bearings) != 2:
        raise ValueError("exactly two source bearings are required")
    if not all(np.isfinite(value) for value in bearings):
        raise ValueError("bearings_deg must be finite")
    if bearings[0] == bearings[1]:
        raise ValueError("the two source bearings must differ")

    if amplitudes is None:
        gains = (1.0, 1.0)
    else:
        gains = tuple(float(value) for value in amplitudes)
        if len(gains) != 2:
            raise ValueError("amplitudes must contain exactly two values")
        if not all(np.isfinite(value) and value >= 0.0 for value in gains):
            raise ValueError("amplitudes must be non-negative and finite")

    length = int(round(float(duration_s) * sample_rate))
    if length < 2:
        raise ValueError("duration_s and sample_rate must produce at least two samples")
    generator = rng if rng is not None else np.random.default_rng()

    source_signals = [_unit_rms_source(length, generator) for _ in range(2)]
    arrival_times = tuple(
        plane_wave_arrival_times(
            positions,
            bearing,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
        for bearing in bearings
    )

    channels = {name: np.zeros(length, dtype=np.float64) for name in positions}
    for source_index in range(2):
        for name in positions:
            contribution = fractional_delay(
                source_signals[source_index],
                arrival_times[source_index][name],
                sample_rate,
            )
            channels[name] += gains[source_index] * contribution

    for name in positions:
        channels[name] += _noise(length, generator, noise_std)

    return TwoSourceSignals(
        signals=channels,
        sample_rate=sample_rate,
        speed_of_sound_m_s=float(speed_of_sound_m_s),
        bearings_deg=bearings,
        amplitudes=gains,
        arrival_times_s=arrival_times,
    )
