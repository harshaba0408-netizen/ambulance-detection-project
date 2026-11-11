from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ambulance_system.localisation as localisation
from ambulance_system.localisation import (
    DelayEstimate,
    DoAError,
    GccPhatError,
    SPEED_OF_SOUND_M_S,
    TdoaObservation,
    estimate_array_bearing,
    estimate_delay,
    estimate_delays,
    estimate_pair_bearing,
    fractional_delay,
    gcc_phat_correlation,
    plane_wave_arrival_times,
    simulate_plane_wave,
    simulate_two_sources,
)

FS = 16000
SPACING = 0.12
PAIR = {"L": (-SPACING / 2.0, 0.0), "R": (SPACING / 2.0, 0.0)}


def _noise(length: int = 3200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(length)


# ---------------------------------------------------------------------------
# GCC-PHAT delay estimation
# ---------------------------------------------------------------------------


def test_integer_delay_is_recovered():
    reference = _noise()
    moving = np.roll(reference, 7)

    estimate = estimate_delay(reference, moving, FS, interpolation=8)

    assert isinstance(estimate, DelayEstimate)
    assert estimate.delay_s == pytest.approx(7.0 / FS, abs=0.05 / FS)
    assert estimate.lag_samples == pytest.approx(7.0, abs=0.05)


def test_sign_convention_is_positive_when_moving_arrives_later():
    reference = _noise(seed=1)
    later = fractional_delay(reference, 2.5 / FS, FS)

    forward = estimate_delay(reference, later, FS, interpolation=16)
    backward = estimate_delay(later, reference, FS, interpolation=16)

    assert forward.delay_s > 0.0
    assert backward.delay_s < 0.0
    assert forward.delay_s == pytest.approx(2.5 / FS, abs=0.05 / FS)
    assert backward.delay_s == pytest.approx(-forward.delay_s, abs=0.05 / FS)


def test_fractional_delay_is_resolved_beyond_one_sample():
    reference = _noise(seed=2, length=6400)
    moving = fractional_delay(reference, 2.5 / FS, FS)

    estimate = estimate_delay(reference, moving, FS, interpolation=16)

    assert abs(estimate.lag_samples - 2.5) < 0.05


def test_zero_delay_peaks_at_zero():
    reference = _noise(seed=3)

    estimate = estimate_delay(reference, reference, FS, interpolation=4)

    assert estimate.delay_s == pytest.approx(0.0, abs=1e-9)
    assert estimate.peak > 0.5


def test_correlation_axis_is_sorted_and_normalised():
    reference = _noise(seed=4, length=1024)
    moving = np.roll(reference, -3)

    delays, values = gcc_phat_correlation(
        reference, moving, FS, max_delay_s=5.0 / FS, interpolation=4
    )

    assert delays.shape == values.shape
    assert np.all(np.diff(delays) > 0.0)
    assert float(np.max(np.abs(values))) <= 1.5
    peak_delay = float(delays[int(np.argmax(np.abs(values)))])
    assert peak_delay == pytest.approx(-3.0 / FS, abs=0.1 / FS)


def test_max_delay_window_limits_the_search():
    reference = _noise(seed=5, length=3200)
    moving = np.roll(reference, 10)

    clipped = estimate_delay(reference, moving, FS, max_delay_s=5.0 / FS)

    assert abs(clipped.delay_s) <= 5.0 / FS + 1e-12

    recoverable = np.roll(reference, 3)
    inside = estimate_delay(reference, recoverable, FS, max_delay_s=5.0 / FS)
    assert inside.delay_s == pytest.approx(3.0 / FS, abs=0.1 / FS)


def test_invalid_gcc_inputs_raise():
    signal = _noise(seed=6, length=64)

    with pytest.raises(GccPhatError):
        estimate_delay(np.zeros((2, 8)), signal, FS)
    with pytest.raises(GccPhatError):
        estimate_delay([1.0], signal, FS)
    with pytest.raises(GccPhatError):
        estimate_delay(np.zeros(64), signal, FS)
    with pytest.raises(GccPhatError):
        estimate_delay(np.where(np.arange(64) == 3, np.nan, signal), signal, FS)
    with pytest.raises(GccPhatError):
        estimate_delay(signal, signal, 0)
    with pytest.raises(GccPhatError):
        estimate_delay(signal, signal, FS, interpolation=0)
    with pytest.raises(GccPhatError):
        estimate_delay(signal, signal, FS, max_delay_s=-0.1)
    with pytest.raises(GccPhatError):
        estimate_delays(signal, signal, FS, max_peaks=0)
    with pytest.raises(GccPhatError):
        estimate_delays(signal, signal, FS, min_separation_s=-1.0)


def test_two_synthetic_sources_produce_two_delay_peaks():
    bearings = (-40.0, 35.0)
    simulated = simulate_two_sources(
        PAIR,
        bearings,
        duration_s=0.5,
        sample_rate=FS,
        noise_std=0.02,
        rng=np.random.default_rng(7),
    )

    estimates = estimate_delays(
        simulated.signals["L"],
        simulated.signals["R"],
        FS,
        max_peaks=2,
        min_separation_s=0.5 / FS,
        interpolation=8,
    )

    expected = sorted(
        simulated.delay_between("L", "R", source) for source in range(2)
    )
    measured = [estimate.delay_s for estimate in estimates]
    assert len(measured) == 2
    for found, target in zip(measured, expected):
        assert found == pytest.approx(target, abs=0.25 / FS)


# ---------------------------------------------------------------------------
# Direction of arrival
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bearing_deg", [-75.0, -40.0, 0.0, 25.0, 60.0])
def test_pair_bearing_round_trip(bearing_deg):
    delay = -SPACING * np.sin(np.radians(bearing_deg)) / SPEED_OF_SOUND_M_S

    estimate = estimate_pair_bearing(delay, SPACING)

    assert estimate.bearing_deg == pytest.approx(bearing_deg, abs=1e-6)
    assert estimate.clipped is False


def test_broadside_delay_gives_zero_bearing():
    estimate = estimate_pair_bearing(0.0, SPACING)

    assert estimate.bearing_deg == pytest.approx(0.0, abs=1e-12)


def test_impossible_pair_delay_is_rejected_or_clipped():
    too_large = 10.0 * SPACING / SPEED_OF_SOUND_M_S

    with pytest.raises(DoAError):
        estimate_pair_bearing(too_large, SPACING, clip=False)

    clipped = estimate_pair_bearing(too_large, SPACING)
    assert clipped.clipped is True
    assert clipped.bearing_deg == pytest.approx(-90.0)


def test_pair_bearing_validates_geometry():
    with pytest.raises(DoAError):
        estimate_pair_bearing(0.0, 0.0)
    with pytest.raises(DoAError):
        estimate_pair_bearing(0.0, SPACING, speed_of_sound_m_s=0.0)
    with pytest.raises(DoAError):
        estimate_pair_bearing(float("nan"), SPACING)


def test_array_solver_recovers_bearing_from_exact_delays():
    positions = {"m0": (0.0, 0.0), "m1": (0.2, 0.0), "m2": (0.0, 0.2)}
    bearing_deg = 35.0
    times = plane_wave_arrival_times(positions, bearing_deg)
    observations = [
        TdoaObservation("m0", "m1", times["m1"] - times["m0"]),
        TdoaObservation("m0", "m2", times["m2"] - times["m0"]),
        TdoaObservation("m1", "m2", times["m2"] - times["m1"]),
    ]

    estimate = estimate_array_bearing(positions, observations)

    assert estimate.bearing_deg == pytest.approx(bearing_deg, abs=1e-6)
    assert estimate.residual_rms_s < 1e-12
    assert estimate.observations == 3
    assert estimate.unit_vector[0] == pytest.approx(np.sin(np.radians(35.0)), abs=1e-6)


def test_array_solver_recovers_bearing_from_synthetic_signals():
    positions = {"m0": (0.0, 0.0), "m1": (0.2, 0.0), "m2": (0.0, 0.2)}
    simulated = simulate_plane_wave(
        positions,
        35.0,
        duration_s=0.5,
        sample_rate=FS,
        noise_std=0.005,
        rng=np.random.default_rng(8),
    )

    def observation(a: str, b: str) -> TdoaObservation:
        return TdoaObservation(
            a,
            b,
            estimate_delay(
                simulated.signals[a], simulated.signals[b], FS, interpolation=16
            ).delay_s,
        )

    estimate = estimate_array_bearing(
        positions,
        [observation("m0", "m1"), observation("m0", "m2"), observation("m1", "m2")],
    )

    assert estimate.bearing_deg == pytest.approx(35.0, abs=1.0)
    assert estimate.residual_rms_s < 5e-5


def test_collinear_array_cannot_resolve_bearing():
    positions = {"a": (0.0, 0.0), "b": (0.1, 0.0), "c": (0.2, 0.0)}
    observations = [
        TdoaObservation("a", "b", 1e-4),
        TdoaObservation("b", "c", 1e-4),
    ]

    with pytest.raises(DoAError, match="collinear"):
        estimate_array_bearing(positions, observations)


def test_array_solver_validates_observations():
    positions = {"a": (0.0, 0.0), "b": (0.1, 0.0), "c": (0.0, 0.1)}

    with pytest.raises(DoAError, match="at least two"):
        estimate_array_bearing(positions, [TdoaObservation("a", "b", 0.0)])

    with pytest.raises(DoAError, match="unknown microphone"):
        estimate_array_bearing(
            positions,
            [
                TdoaObservation("a", "b", 0.0),
                TdoaObservation("a", "z", 0.0),
            ],
        )

    with pytest.raises(DoAError, match="two different"):
        TdoaObservation("a", "a", 0.0)

    with pytest.raises(DoAError, match="non-empty"):
        TdoaObservation("", "b", 0.0)


# ---------------------------------------------------------------------------
# Synthetic array signals
# ---------------------------------------------------------------------------


def test_fractional_delay_matches_integer_roll():
    signal = _noise(seed=9, length=1024)
    shifted = fractional_delay(signal, 3.0 / FS, FS)

    assert np.allclose(shifted, np.roll(signal, 3), atol=1e-9)


def test_broadside_arrival_times_are_equal_and_endfire_has_opposite_signs():
    broadside = plane_wave_arrival_times(PAIR, 0.0)
    assert broadside["L"] == pytest.approx(broadside["R"], abs=1e-15)

    endfire = plane_wave_arrival_times(PAIR, 90.0)
    assert endfire["R"] < endfire["L"]


def test_simulated_pair_matches_the_ground_truth_delay():
    simulated = simulate_plane_wave(
        PAIR,
        25.0,
        duration_s=0.25,
        sample_rate=FS,
        noise_std=0.01,
        rng=np.random.default_rng(10),
    )

    expected = simulated.delay_between("L", "R")
    estimate = estimate_delay(
        simulated.signals["L"], simulated.signals["R"], FS, interpolation=16
    )
    bearing = estimate_pair_bearing(estimate.delay_s, SPACING)

    assert estimate.delay_s == pytest.approx(expected, abs=0.1 / FS)
    assert bearing.bearing_deg == pytest.approx(25.0, abs=0.5)
    assert bearing.clipped is False


def test_simulation_is_reproducible_with_a_seed():
    first = simulate_plane_wave(PAIR, 10.0, rng=np.random.default_rng(11))
    second = simulate_plane_wave(PAIR, 10.0, rng=np.random.default_rng(11))

    assert np.array_equal(first.signals["L"], second.signals["L"])
    assert np.array_equal(first.signals["R"], second.signals["R"])


def test_two_source_simulation_reports_per_source_delays():
    simulated = simulate_two_sources(
        PAIR,
        (-30.0, 30.0),
        amplitudes=(1.0, 0.5),
        duration_s=0.1,
        sample_rate=FS,
        rng=np.random.default_rng(12),
    )

    assert simulated.bearings_deg == (-30.0, 30.0)
    assert simulated.amplitudes == (1.0, 0.5)
    assert simulated.delay_between("L", "R", 0) == pytest.approx(
        SPACING * np.sin(np.radians(30.0)) / SPEED_OF_SOUND_M_S, abs=1e-12
    )
    assert simulated.delay_between("L", "R", 1) == pytest.approx(
        -SPACING * np.sin(np.radians(30.0)) / SPEED_OF_SOUND_M_S, abs=1e-12
    )


def test_simulation_validates_inputs():
    with pytest.raises(ValueError):
        simulate_plane_wave({}, 0.0)
    with pytest.raises(ValueError):
        simulate_plane_wave({"a": (0.0, 0.0, 0.0)}, 0.0)
    with pytest.raises(ValueError):
        simulate_plane_wave(PAIR, 0.0, duration_s=0.0)
    with pytest.raises(ValueError):
        simulate_plane_wave(PAIR, 0.0, sample_rate=0)
    with pytest.raises(ValueError):
        simulate_plane_wave(PAIR, 0.0, source=np.ones(4))
    with pytest.raises(ValueError):
        simulate_two_sources(PAIR, (10.0,))
    with pytest.raises(ValueError):
        simulate_two_sources(PAIR, (10.0, 10.0))
    with pytest.raises(ValueError):
        simulate_two_sources(PAIR, (10.0, 20.0), amplitudes=(1.0,))


def test_localisation_package_imports_no_decision_or_device_module():
    package = Path(localisation.__file__).parent
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )

    for forbidden in (
        "ambulance_system.fusion",
        "ambulance_system.traffic_signal",
        "ambulance_system.dashboard",
        "ambulance_system.hardware",
        "import sounddevice",
        "import pyaudio",
        "import serial",
        "sklearn",
    ):
        assert forbidden not in source
