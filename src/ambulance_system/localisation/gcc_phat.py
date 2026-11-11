"""GCC-PHAT time-difference-of-arrival estimation.

This is a self-contained offline research module. It estimates the delay
between two synchronised channels of one signal; it does **not** classify
sirens, decide a priority phase or claim a geographic position from a single
microphone.

Method
------
For two real signals ``a[n]`` (reference) and ``b[n]`` (moving), the
generalised cross-correlation with phase transform is

.. math::

    R_{ab}(f) = \\frac{A(f) B^{*}(f)}{|A(f) B^{*}(f)|}

and its inverse Fourier transform is the GCC-PHAT correlation. Dividing by the
magnitude discards the source spectrum and keeps only the phase difference, so
the peak stays sharp even when the source is not white. The delay is the
location of that peak:

.. math::

    \\tau = \\arg\\max_{\\tau} \\left| \\mathrm{IFFT}\\{R_{ab}\\}(\\tau) \\right|

A positive result means ``moving`` arrives **later** than ``reference``.

Only NumPy is used. No audio device, classifier or decision module is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


class GccPhatError(ValueError):
    """Raised for invalid GCC-PHAT inputs."""


@dataclass(frozen=True)
class DelayEstimate:
    """One time-difference-of-arrival measurement.

    Attributes:
        delay_s: Time by which ``moving`` lags ``reference``, in seconds.
            Positive means ``moving`` arrives later.
        lag_samples: The same delay expressed in samples at the analysis rate.
        peak: Absolute height of the normalised GCC-PHAT peak. It is a relative
            sharpness indicator, **not** a calibrated probability or confidence.
    """

    delay_s: float
    lag_samples: float
    peak: float

    def to_dict(self) -> dict[str, float]:
        return {
            "delay_s": round(float(self.delay_s), 9),
            "lag_samples": round(float(self.lag_samples), 6),
            "peak": round(float(self.peak), 6),
        }


def _as_signal(values: Sequence[float], name: str) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise GccPhatError(f"{name} must be one-dimensional")
    if signal.size < 2:
        raise GccPhatError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(signal)):
        raise GccPhatError(f"{name} contains NaN or infinite values")
    signal = signal - float(np.mean(signal))
    scale = float(np.std(signal))
    if scale <= 0.0:
        raise GccPhatError(f"{name} has zero variance")
    return signal / scale


def _validate(sample_rate: float, interpolation: int, max_delay_s: Optional[float]) -> None:
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise GccPhatError("sample_rate must be a positive finite number")
    if isinstance(interpolation, bool) or not isinstance(interpolation, (int, np.integer)):
        raise GccPhatError("interpolation must be a positive integer")
    if interpolation < 1:
        raise GccPhatError("interpolation must be a positive integer")
    if max_delay_s is not None:
        if not np.isfinite(max_delay_s) or max_delay_s <= 0.0:
            raise GccPhatError("max_delay_s must be positive and finite when provided")


def gcc_phat_correlation(
    reference: Sequence[float],
    moving: Sequence[float],
    sample_rate: float,
    *,
    max_delay_s: Optional[float] = None,
    interpolation: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the GCC-PHAT correlation as ``(delays_s, values)``.

    Args:
        reference: Reference channel samples.
        moving: Channel whose delay relative to ``reference`` is measured.
        sample_rate: Sample rate of both channels in Hz. The channels must be
            sampled synchronously at the same rate.
        max_delay_s: Optional search bound in seconds. When omitted the full
            non-circular range is returned. Values beyond the physical range of
            a microphone pair should be excluded by the caller.
        interpolation: IFFT oversampling factor (1, 2, 4, ...). Higher values
            give a finer peak grid without changing the analysed band.

    Returns:
        A tuple of two one-dimensional arrays sorted by increasing delay. The
        second array is the real GCC-PHAT correlation, normalised so that a
        perfectly aligned unit-variance pair peaks near 1.0.
    """
    _validate(sample_rate, interpolation, max_delay_s)
    a = _as_signal(reference, "reference")
    b = _as_signal(moving, "moving")

    # Linear, not circular, correlation: zero-pad to len(a) + len(b).
    n_fft = a.size + b.size
    spectrum = np.fft.rfft(a, n=n_fft) * np.conj(np.fft.rfft(b, n=n_fft))
    magnitude = np.abs(spectrum)

    # PHAT weighting. Leave true zero bins at zero instead of amplifying noise.
    weights = np.divide(
        spectrum,
        magnitude,
        out=np.zeros_like(spectrum),
        where=magnitude > np.finfo(np.float64).eps,
    )

    oversampled_length = n_fft * int(interpolation)
    correlation = np.fft.irfft(weights, n=oversampled_length)
    # Zero-padding the spectrum divides the time-domain peak by the
    # interpolation factor. Undo that so `peak` means the same thing at every
    # interpolation setting (a perfect match is about 1.0).
    correlation *= float(interpolation)

    half = oversampled_length // 2
    max_shift = half
    if max_delay_s is not None:
        requested = int(np.floor(float(max_delay_s) * sample_rate * interpolation))
        max_shift = min(requested, half)

    lags = np.arange(-max_shift, max_shift + 1, dtype=np.int64)
    values = correlation[lags % oversampled_length]
    delays = -lags.astype(np.float64) / (float(sample_rate) * interpolation)

    order = np.argsort(delays)
    return delays[order], values[order]


def estimate_delays(
    reference: Sequence[float],
    moving: Sequence[float],
    sample_rate: float,
    *,
    max_delay_s: Optional[float] = None,
    interpolation: int = 16,
    max_peaks: int = 1,
    min_separation_s: Optional[float] = None,
) -> Tuple[DelayEstimate, ...]:
    """Estimate one or more TDOA peaks with non-maximum suppression.

    Multiple peaks are useful only as a diagnostic for multi-source scenes. A
    conventional GCC-PHAT estimate is the single strongest peak; with two
    simultaneous sources the method can return both local maxima, but it cannot
    assign a source identity or resolve every overlap.

    Args:
        reference: Reference channel samples.
        moving: Channel whose delay relative to ``reference`` is measured.
        sample_rate: Sample rate in Hz.
        max_delay_s: Optional physical search bound in seconds.
        interpolation: IFFT oversampling factor.
        max_peaks: Number of peaks to return (1 by default).
        min_separation_s: Minimum spacing between reported peaks. Defaults to
            one native sample, which suppresses adjacent oversampled bins of
            the same peak.

    Returns:
        A tuple of :class:`DelayEstimate`, ordered by increasing delay.
    """
    if isinstance(max_peaks, bool) or not isinstance(max_peaks, (int, np.integer)):
        raise GccPhatError("max_peaks must be a positive integer")
    if max_peaks < 1:
        raise GccPhatError("max_peaks must be a positive integer")

    _validate(sample_rate, interpolation, max_delay_s)

    if min_separation_s is None:
        min_separation_s = 1.0 / float(sample_rate)
    if not np.isfinite(min_separation_s) or min_separation_s < 0.0:
        raise GccPhatError("min_separation_s must be non-negative and finite")

    delays, values = gcc_phat_correlation(
        reference,
        moving,
        sample_rate,
        max_delay_s=max_delay_s,
        interpolation=interpolation,
    )
    magnitude = np.abs(values)
    bin_width_s = (delays[1] - delays[0]) if delays.size > 1 else 1.0 / sample_rate
    guard_bins = int(np.ceil(min_separation_s / abs(bin_width_s))) if bin_width_s else 0

    order = np.argsort(magnitude)[::-1]
    chosen: list[int] = []
    for index in order:
        if all(abs(int(index) - previous) > guard_bins for previous in chosen):
            chosen.append(int(index))
        if len(chosen) >= max_peaks:
            break
    if not chosen:
        raise GccPhatError("correlation contains no usable peak")

    estimates = [
        DelayEstimate(
            delay_s=float(delays[index]),
            lag_samples=float(delays[index] * sample_rate),
            peak=float(magnitude[index]),
        )
        for index in chosen
    ]
    estimates.sort(key=lambda estimate: estimate.delay_s)
    return tuple(estimates)


def estimate_delay(
    reference: Sequence[float],
    moving: Sequence[float],
    sample_rate: float,
    *,
    max_delay_s: Optional[float] = None,
    interpolation: int = 16,
) -> DelayEstimate:
    """Estimate the single strongest TDOA between two channels.

    This is the convenience wrapper used by :mod:`ambulance_system.localisation.doa`.
    """
    return estimate_delays(
        reference,
        moving,
        sample_rate,
        max_delay_s=max_delay_s,
        interpolation=interpolation,
        max_peaks=1,
    )[0]
