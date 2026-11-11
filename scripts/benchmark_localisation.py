"""Synthetic GCC-PHAT localisation benchmark (Stage 13).

This script generates free-field two-microphone signals with known delays and
measures the accuracy of the offline GCC-PHAT estimator and bearing solver. It
does **not** use a real microphone array, a siren recording or any priority
logic. The output is a synthetic method check, not a field measurement.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\benchmark_localisation.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\benchmark_localisation.py
    .\\venv\\Scripts\\python.exe scripts\\benchmark_localisation.py --output benchmarks\\results\\localisation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402

from ambulance_system.localisation import (  # noqa: E402
    SPEED_OF_SOUND_M_S,
    estimate_delay,
    estimate_delays,
    estimate_pair_bearing,
    simulate_plane_wave,
    simulate_two_sources,
)

import benchmark_common  # noqa: E402

DEFAULT_BEARINGS = (-75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0)
DEFAULT_NOISE = (0.0, 0.01, 0.05, 0.1)
TWO_SOURCE_BEARINGS = ((-40.0, 35.0), (-60.0, 20.0), (45.0, -50.0))


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": round(float(np.mean(values)), 6),
        "p95": round(float(benchmark_common.percentile(values, 0.95)), 6),
        "max": round(float(np.max(values)), 6),
    }


def run_single_source(
    *,
    bearings: tuple[float, ...],
    noise_levels: tuple[float, ...],
    trials: int,
    spacing_m: float,
    sample_rate: int,
    duration_s: float,
    interpolation: int,
    seed: int,
) -> dict[str, Any]:
    positions = {"L": (-spacing_m / 2.0, 0.0), "R": (spacing_m / 2.0, 0.0)}
    by_noise: list[dict[str, Any]] = []
    all_delay_errors_us: list[float] = []
    all_bearing_errors_deg: list[float] = []
    all_within_2deg: list[bool] = []

    for noise_std in noise_levels:
        delay_errors_us: list[float] = []
        bearing_errors_deg: list[float] = []
        within_2deg: list[bool] = []
        clipped = 0
        for bearing_deg in bearings:
            for trial in range(trials):
                rng = np.random.default_rng(seed + trial + int(round(bearing_deg * 100)))
                simulated = simulate_plane_wave(
                    positions,
                    bearing_deg,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    noise_std=noise_std,
                    rng=rng,
                )
                truth_s = simulated.delay_between("L", "R")
                estimate = estimate_delay(
                    simulated.signals["L"],
                    simulated.signals["R"],
                    sample_rate,
                    interpolation=interpolation,
                )
                delay_errors_us.append(abs(estimate.delay_s - truth_s) * 1e6)
                bearing = estimate_pair_bearing(estimate.delay_s, spacing_m)
                clipped += int(bearing.clipped)
                error = abs(bearing.bearing_deg - bearing_deg)
                bearing_errors_deg.append(error)
                within_2deg.append(error <= 2.0)

        all_delay_errors_us.extend(delay_errors_us)
        all_bearing_errors_deg.extend(bearing_errors_deg)
        all_within_2deg.extend(within_2deg)
        by_noise.append(
            {
                "noise_std": noise_std,
                "trials": len(delay_errors_us),
                "delay_error_us": _summary(delay_errors_us),
                "bearing_error_deg": _summary(bearing_errors_deg),
                "within_2deg_rate": round(float(np.mean(within_2deg)), 6),
                "clipped": clipped,
            }
        )

    return {
        "bearings_deg": list(bearings),
        "noise_levels": list(noise_levels),
        "trials_per_condition": trials,
        "by_noise_std": by_noise,
        "overall": {
            "trials": len(all_delay_errors_us),
            "delay_error_us": _summary(all_delay_errors_us),
            "bearing_error_deg": _summary(all_bearing_errors_deg),
            "within_2deg_rate": round(float(np.mean(all_within_2deg)), 6),
        },
    }


def run_two_source(
    *,
    spacing_m: float,
    sample_rate: int,
    duration_s: float,
    interpolation: int,
    trials: int,
    seed: int,
    noise_std: float = 0.01,
    tolerance_samples: float = 0.5,
) -> dict[str, Any]:
    positions = {"L": (-spacing_m / 2.0, 0.0), "R": (spacing_m / 2.0, 0.0)}
    conditions: list[dict[str, Any]] = []
    detected = 0
    total = 0
    matched_errors_us: list[float] = []

    for bearings in TWO_SOURCE_BEARINGS:
        condition_detected = 0
        condition_total = 0
        condition_errors: list[float] = []
        for trial in range(trials):
            total += 1
            condition_total += 1
            simulated = simulate_two_sources(
                positions,
                bearings,
                duration_s=duration_s,
                sample_rate=sample_rate,
                noise_std=noise_std,
                rng=np.random.default_rng(seed + trial + int(round(bearings[0] * 10))),
            )
            truth = sorted(
                simulated.delay_between("L", "R", source) for source in range(2)
            )
            estimates = estimate_delays(
                simulated.signals["L"],
                simulated.signals["R"],
                sample_rate,
                interpolation=interpolation,
                max_peaks=2,
                min_separation_s=0.5 / sample_rate,
            )
            measured = [estimate.delay_s for estimate in estimates]
            tolerance_s = tolerance_samples / sample_rate
            if len(measured) == 2 and all(
                min(abs(found - target) for found in measured) <= tolerance_s
                for target in truth
            ):
                detected += 1
                condition_detected += 1
                nearest = [
                    min(abs(found - target) for found in measured) for target in truth
                ]
                condition_errors.extend(error * 1e6 for error in nearest)
                matched_errors_us.extend(error * 1e6 for error in nearest)
        conditions.append(
            {
                "bearings_deg": list(bearings),
                "trials": condition_total,
                "both_sources_detected": condition_detected,
                "detection_rate": round(condition_detected / condition_total, 6),
                "matched_delay_error_us": _summary(condition_errors),
            }
        )

    return {
        "conditions": conditions,
        "trials": total,
        "both_sources_detected": detected,
        "detection_rate": round(detected / total, 6) if total else None,
        "tolerance_samples": tolerance_samples,
        "noise_std": noise_std,
        "matched_delay_error_us": _summary(matched_errors_us),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs", type=int, default=16000, help="sample rate in Hz")
    parser.add_argument("--duration-s", type=float, default=0.25, help="signal length")
    parser.add_argument("--spacing-m", type=float, default=0.12, help="microphone spacing")
    parser.add_argument("--interpolation", type=int, default=16, help="IFFT oversampling")
    parser.add_argument("--trials", type=int, default=10, help="trials per condition")
    parser.add_argument(
        "--two-source-trials", type=int, default=5, help="trials per two-source condition"
    )
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print configuration and synthetic bounds without running",
    )
    return parser


def base_payload(args: argparse.Namespace) -> dict[str, Any]:
    max_delay_s = args.spacing_m / SPEED_OF_SOUND_M_S
    return {
        "benchmark": "localisation",
        "kind": "synthetic_gcc_phat_method_check",
        "generated_at": benchmark_common.iso_now(),
        "command": sys.argv,
        "environment": benchmark_common.environment(),
        "microphone_array_used": False,
        "real_audio_used": False,
        "priority_decision_connected": False,
        "method": {
            "estimator": "GCC-PHAT",
            "doa_model": "far-field plane wave, 1-D pair bearing",
            "signal_model": "white source, independent Gaussian sensor noise",
            "fractional_delay": "FFT phase shift (circular within the window)",
            "speed_of_sound_m_s": SPEED_OF_SOUND_M_S,
        },
        "configuration": {
            "sample_rate": args.fs,
            "duration_s": args.duration_s,
            "spacing_m": args.spacing_m,
            "interpolation": args.interpolation,
            "trials": args.trials,
            "two_source_trials": args.two_source_trials,
            "seed": args.seed,
            "bearings_deg": list(DEFAULT_BEARINGS),
            "noise_levels": list(DEFAULT_NOISE),
            "two_source_bearings_deg": [list(pair) for pair in TWO_SOURCE_BEARINGS],
        },
        "synthetic_bounds": {
            "max_unambiguous_delay_s": round(max_delay_s, 9),
            "max_unambiguous_delay_samples": round(max_delay_s * args.fs, 6),
            "one_sample_s": round(1.0 / args.fs, 9),
        },
        "caveats": [
            "Synthetic free-field signals only; no microphone array or real recording was used.",
            "No reverberation, walls, sensor directivity, gain mismatch or clock drift is modelled.",
            "The bearing is a far-field angle from broadside for a linear pair, not a geographic position.",
            "The experiment is not connected to fusion, signal priority or the dashboard.",
            "These numbers must not be quoted as field localisation accuracy.",
        ],
    }


def default_output_path() -> Path:
    stamp = benchmark_common.utc_timestamp()
    return ROOT / "benchmarks" / "results" / f"localisation_synthetic_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fs <= 0 or args.duration_s <= 0.0 or args.spacing_m <= 0.0:
        raise ValueError("fs, duration_s and spacing_m must be positive")
    if args.interpolation < 1:
        raise ValueError("interpolation must be at least 1")
    if args.trials < 1 or args.two_source_trials < 1:
        raise ValueError("trial counts must be positive")

    payload = base_payload(args)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    payload["results"] = {
        "single_source": run_single_source(
            bearings=DEFAULT_BEARINGS,
            noise_levels=DEFAULT_NOISE,
            trials=args.trials,
            spacing_m=args.spacing_m,
            sample_rate=args.fs,
            duration_s=args.duration_s,
            interpolation=args.interpolation,
            seed=args.seed,
        ),
        "two_source": run_two_source(
            spacing_m=args.spacing_m,
            sample_rate=args.fs,
            duration_s=max(args.duration_s, 0.5),
            interpolation=args.interpolation,
            trials=args.two_source_trials,
            seed=args.seed,
        ),
    }

    output = Path(args.output) if args.output else default_output_path()
    payload["output"] = str(output)
    benchmark_common.write_json(payload, output)

    overall = payload["results"]["single_source"]["overall"]
    print(
        "single-source synthetic: "
        f"{overall['trials']} trials, "
        f"delay MAE {overall['delay_error_us']['mean']:.2f} us, "
        f"bearing MAE {overall['bearing_error_deg']['mean']:.3f} deg, "
        f"within 2 deg {overall['within_2deg_rate'] * 100:.1f}%"
    )
    for row in payload["results"]["single_source"]["by_noise_std"]:
        print(
            f"  noise_std={row['noise_std']:.3f}: "
            f"delay MAE {row['delay_error_us']['mean']:.2f} us, "
            f"bearing MAE {row['bearing_error_deg']['mean']:.3f} deg"
        )
    two = payload["results"]["two_source"]
    print(
        "two-source synthetic: "
        f"{two['both_sources_detected']}/{two['trials']} both peaks detected "
        f"({(two['detection_rate'] or 0.0) * 100:.1f}%)"
    )
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
