"""Far-field direction-of-arrival (DoA) from time-difference measurements.

The geometry in this module is deliberately simple and explicit:

* sensors are points in a metric two-dimensional plane;
* a source is far enough away that its wavefront is planar across the array;
* the source bearing ``theta`` is measured from broadside (``+y``) towards
  ``+x``, so ``theta = +90 deg`` points along ``+x``;
* a measurement ``TdoaObservation(mic_a, mic_b, delay_s)`` means
  ``delay_s = t(mic_b) - t(mic_a)``.

For a unit vector ``u = (sin(theta), cos(theta))`` pointing from the array
towards the source, a sensor at ``p`` receives the wave at

.. math::

    t(p) = t_0 - \\frac{p \\cdot u}{c}

Therefore each pair gives one linear equation

.. math::

    (p_b - p_a) \\cdot u = -c \\, \\tau_{ab}

Two or more non-collinear pairs determine ``u``. The solver normalises the
least-squares vector and reports the residual in seconds.

This is offline research code. It does not fuse direction with siren
classification, does not decide a traffic phase and cannot locate a source from
a single microphone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

#: Nominal speed of sound in dry air at about 20 degrees Celsius, in m/s.
SPEED_OF_SOUND_M_S = 343.0


class DoAError(ValueError):
    """Raised for invalid geometry or physically impossible delays."""


@dataclass(frozen=True)
class TdoaObservation:
    """A delay measured between two named sensors.

    ``delay_s`` is positive when the signal reaches ``mic_b`` later than
    ``mic_a``. The convention matches
    :func:`ambulance_system.localisation.gcc_phat.estimate_delay` when
    ``reference=mic_a`` and ``moving=mic_b``.
    """

    mic_a: str
    mic_b: str
    delay_s: float

    def __post_init__(self) -> None:
        if not str(self.mic_a).strip() or not str(self.mic_b).strip():
            raise DoAError("microphone names must be non-empty")
        if self.mic_a == self.mic_b:
            raise DoAError("a TDOA observation needs two different microphones")
        if not np.isfinite(self.delay_s):
            raise DoAError("delay_s must be finite")


@dataclass(frozen=True)
class PairBearingEstimate:
    """Bearing from a single pair of sensors spaced along the x axis."""

    bearing_deg: float
    bearing_rad: float
    delay_s: float
    sensor_spacing_m: float
    speed_of_sound_m_s: float
    clipped: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "bearing_deg": round(float(self.bearing_deg), 4),
            "bearing_rad": round(float(self.bearing_rad), 8),
            "delay_s": round(float(self.delay_s), 9),
            "sensor_spacing_m": round(float(self.sensor_spacing_m), 6),
            "speed_of_sound_m_s": round(float(self.speed_of_sound_m_s), 3),
            "clipped": bool(self.clipped),
        }


@dataclass(frozen=True)
class ArrayBearingEstimate:
    """Bearing solved from several TDOA observations on a 2-D array."""

    bearing_deg: float
    bearing_rad: float
    unit_vector: Tuple[float, float]
    residual_rms_s: float
    observations: int
    speed_of_sound_m_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "bearing_deg": round(float(self.bearing_deg), 4),
            "bearing_rad": round(float(self.bearing_rad), 8),
            "unit_vector": [round(float(value), 6) for value in self.unit_vector],
            "residual_rms_s": round(float(self.residual_rms_s), 9),
            "observations": int(self.observations),
            "speed_of_sound_m_s": round(float(self.speed_of_sound_m_s), 3),
        }


def estimate_pair_bearing(
    delay_s: float,
    sensor_spacing_m: float,
    *,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
    clip: bool = True,
) -> PairBearingEstimate:
    """Convert one pair delay into a bearing.

    The microphones are placed at ``(-spacing/2, 0)`` and ``(+spacing/2, 0)``.
    ``delay_s`` must be ``t(+x sensor) - t(-x sensor)``. A source towards ``+x``
    reaches the ``+x`` sensor first, so its delay is negative and its bearing is
    positive.

    Args:
        delay_s: Arrival-time difference in seconds under the convention above.
        sensor_spacing_m: Distance between the two sensors in metres.
        speed_of_sound_m_s: Assumed propagation speed in m/s.
        clip: When True, an out-of-range delay is clamped to the endfire
            direction and ``clipped`` is set. When False, the function raises.

    Returns:
        A :class:`PairBearingEstimate` with ``bearing_deg`` in ``[-90, 90]``.
    """
    if not np.isfinite(sensor_spacing_m) or sensor_spacing_m <= 0.0:
        raise DoAError("sensor_spacing_m must be positive and finite")
    if not np.isfinite(speed_of_sound_m_s) or speed_of_sound_m_s <= 0.0:
        raise DoAError("speed_of_sound_m_s must be positive and finite")
    if not np.isfinite(delay_s):
        raise DoAError("delay_s must be finite")

    ratio = -float(speed_of_sound_m_s) * float(delay_s) / float(sensor_spacing_m)
    clipped = bool(abs(ratio) > 1.0)
    if clipped and not clip:
        raise DoAError(
            "delay exceeds the physical limit of this pair; "
            "check the microphone order or spacing"
        )
    ratio = float(np.clip(ratio, -1.0, 1.0))
    bearing_rad = float(np.arcsin(ratio))
    return PairBearingEstimate(
        bearing_deg=float(np.degrees(bearing_rad)),
        bearing_rad=bearing_rad,
        delay_s=float(delay_s),
        sensor_spacing_m=float(sensor_spacing_m),
        speed_of_sound_m_s=float(speed_of_sound_m_s),
        clipped=clipped,
    )


def _validated_positions(
    positions_m: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    if len(positions_m) < 2:
        raise DoAError("at least two named sensor positions are required")
    validated: dict[str, np.ndarray] = {}
    for name, position in positions_m.items():
        label = str(name).strip()
        if not label:
            raise DoAError("sensor names must be non-empty")
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (2,):
            raise DoAError(f"sensor {name!r} must have exactly two coordinates")
        if not np.all(np.isfinite(point)):
            raise DoAError(f"sensor {name!r} has non-finite coordinates")
        validated[label] = point
    return validated


def estimate_array_bearing(
    positions_m: Mapping[str, Sequence[float]],
    observations: Iterable[TdoaObservation],
    *,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> ArrayBearingEstimate:
    """Solve a far-field bearing from two or more non-collinear TDOA pairs.

    The least-squares solution is normalised to a unit vector. A single pair, or
    a set of collinear pairs, cannot resolve two dimensions and raises
    :class:`DoAError`. The reported residual is in seconds and compares the
    measured delays with the delays implied by the fitted bearing.

    Front/back ambiguity is not resolved by a collinear array; use at least
    three non-collinear sensors for a two-dimensional bearing.
    """
    if not np.isfinite(speed_of_sound_m_s) or speed_of_sound_m_s <= 0.0:
        raise DoAError("speed_of_sound_m_s must be positive and finite")

    positions = _validated_positions(positions_m)
    rows: list[list[float]] = []
    targets: list[float] = []
    delays: list[float] = []
    for observation in observations:
        if not isinstance(observation, TdoaObservation):
            raise DoAError("observations must be TdoaObservation instances")
        if observation.mic_a not in positions or observation.mic_b not in positions:
            raise DoAError(
                f"observation references an unknown microphone: "
                f"{observation.mic_a!r}, {observation.mic_b!r}"
            )
        delta = positions[observation.mic_b] - positions[observation.mic_a]
        rows.append([float(delta[0]), float(delta[1])])
        targets.append(-float(speed_of_sound_m_s) * float(observation.delay_s))
        delays.append(float(observation.delay_s))

    if len(rows) < 2:
        raise DoAError("at least two TDOA observations are required")

    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 2:
        raise DoAError(
            "sensor geometry is collinear; add a sensor off the line to resolve bearing"
        )

    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    norm = float(np.linalg.norm(solution))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise DoAError("TDOA observations do not define a direction")
    unit = solution / norm

    predicted_delays = -(design @ unit) / float(speed_of_sound_m_s)
    residual = predicted_delays - np.asarray(delays, dtype=np.float64)
    residual_rms = float(np.sqrt(np.mean(residual**2)))

    bearing_rad = float(np.arctan2(unit[0], unit[1]))
    return ArrayBearingEstimate(
        bearing_deg=float(np.degrees(bearing_rad)),
        bearing_rad=bearing_rad,
        unit_vector=(float(unit[0]), float(unit[1])),
        residual_rms_s=residual_rms,
        observations=len(rows),
        speed_of_sound_m_s=float(speed_of_sound_m_s),
    )
