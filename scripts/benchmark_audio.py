"""Measure the existing siren classifier on every local audio clip.

IMPORTANT: the published Random Forest was trained on these same 400 clips.
The result is therefore **resubstitution**, not held-out validation. It shows
what the shipped model does on its training data; it says nothing about unseen
recordings. The JSON output repeats this caveat next to the metrics.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\benchmark_audio.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\benchmark_audio.py
    .\\venv\\Scripts\\python.exe scripts\\benchmark_audio.py --output benchmarks\\results\\audio.json

The script measures feature extraction and prediction separately, so the
latency table can distinguish file I/O + MFCC cost from model cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from ambulance_system.config import load_config  # noqa: E402

import benchmark_common  # noqa: E402

SIREN_LABEL = 1
NON_SIREN_LABEL = 0


def binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, Optional[float]]:
    """Derive standard binary metrics; undefined ratios stay `None`."""
    total = tp + fp + tn + fn
    precision = benchmark_common.ratio(tp, tp + fp)
    recall = benchmark_common.ratio(tp, tp + fn)
    specificity = benchmark_common.ratio(tn, tn + fp)
    f1 = benchmark_common.ratio(2 * tp, 2 * tp + fp + fn)
    balanced = (
        None
        if recall is None or specificity is None
        else (recall + specificity) / 2.0
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "samples": total,
        "accuracy": benchmark_common.ratio(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": benchmark_common.ratio(fp, fp + tn),
        "false_negative_rate": benchmark_common.ratio(fn, fn + tp),
        "balanced_accuracy": balanced,
    }


def _round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def content_duplicates(paths: list[Path]) -> list[dict[str, Any]]:
    """Group audio files whose bytes are identical.

    Duplicate content across classes is a leakage risk for any future split;
    this harness reports it but never deletes or relabels a file.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[benchmark_common.sha256_file(path)].append(path)
    duplicates = [
        {
            "sha256": digest,
            "files": [_display_path(path) for path in members],
            "classes": sorted({path.parent.name for path in members}),
        }
        for digest, members in groups.items()
        if len(members) > 1
    ]
    return sorted(duplicates, key=lambda item: item["files"][0])


def _display_path(path: Path) -> str:
    """Prefer a repository-relative path, but keep external paths readable."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_files() -> list[tuple[Path, int]]:
    """Return (wav path, true label) in deterministic order."""
    roots = (
        (ROOT / "data" / "audio" / "siren", SIREN_LABEL),
        (ROOT / "data" / "audio" / "non-siren", NON_SIREN_LABEL),
    )
    files: list[tuple[Path, int]] = []
    for directory, label in roots:
        if not directory.is_dir():
            raise FileNotFoundError(f"audio directory not found: {directory}")
        files.extend((path, label) for path in sorted(directory.glob("*.wav")))
    if not files:
        raise FileNotFoundError("no WAV files found under data/audio")
    return files


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=config.model.siren_classifier_path, help="pickled classifier path"
    )
    parser.add_argument("--sample-rate", type=int, default=config.audio.sample_rate)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=config.audio.window_s,
        help="analysis window; the original training code used 3.0",
    )
    parser.add_argument("--threshold", type=float, default=config.audio.threshold)
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list file counts and duplicate groups without loading the model",
    )
    return parser


def base_payload(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"classifier not found: {model_path}")

    files = collect_files()
    siren = sum(1 for _, label in files if label == SIREN_LABEL)
    non_siren = sum(1 for _, label in files if label == NON_SIREN_LABEL)
    duplicates = content_duplicates([path for path, _ in files])

    return {
        "benchmark": "audio",
        "kind": "measured_classifier_evaluation",
        "evaluation_kind": "resubstitution",
        "caveat": (
            "The shipped classifier was trained on these same clips. Metrics describe "
            "training-data fit, not held-out accuracy, and must not be reported as "
            "validation or real-world performance."
        ),
        "generated_at": benchmark_common.iso_now(),
        "command": sys.argv,
        "environment": benchmark_common.environment(),
        "model": {
            "path": str(model_path.resolve()),
            "size_bytes": model_path.stat().st_size,
            "sha256": benchmark_common.sha256_file(model_path),
            "training_unchanged": True,
        },
        "dataset": {
            "root": str((ROOT / "data" / "audio").resolve()),
            "files": len(files),
            "siren": siren,
            "non_siren": non_siren,
            "content_duplicate_groups": duplicates,
            "content_duplicate_group_count": len(duplicates),
        },
        "settings": {
            "sample_rate": args.sample_rate,
            "duration_s": args.duration_s,
            "n_mfcc": None,  # resolved from the loaded classifier/feature code
            "decision_threshold": args.threshold,
        },
    }


def evaluate(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    from ambulance_system.audio.classifier import SirenClassifier
    from ambulance_system.audio.features import load_audio, mfcc_features

    classifier = SirenClassifier.from_path(
        args.model, threshold=args.threshold, sample_rate=args.sample_rate
    )
    payload["settings"]["n_mfcc"] = classifier.n_mfcc

    tp = fp = tn = fn = 0
    label_threshold_mismatches = 0
    load_samples: list[float] = []
    feature_samples: list[float] = []
    predict_samples: list[float] = []
    total_samples: list[float] = []
    durations: list[float] = []
    confidence_values: list[float] = []

    files = collect_files()
    started = time.perf_counter()
    for path, truth in files:
        begin = time.perf_counter()
        samples, _ = load_audio(
            path, sample_rate=args.sample_rate, duration=args.duration_s
        )
        after_load = time.perf_counter()
        features = mfcc_features(
            samples,
            sample_rate=args.sample_rate,
            n_mfcc=classifier.n_mfcc,
        )
        after_features = time.perf_counter()
        prediction = classifier.predict_features(features)
        after_predict = time.perf_counter()

        load_samples.append(after_load - begin)
        feature_samples.append(after_features - after_load)
        predict_samples.append(after_predict - after_features)
        total_samples.append(after_predict - begin)
        durations.append(len(samples) / float(args.sample_rate))
        if prediction.confidence is not None:
            confidence_values.append(float(prediction.confidence))
        if prediction.is_siren != (prediction.label == SIREN_LABEL):
            label_threshold_mismatches += 1

        predicted = prediction.is_siren
        if truth == SIREN_LABEL and predicted:
            tp += 1
        elif truth == SIREN_LABEL and not predicted:
            fn += 1
        elif truth == NON_SIREN_LABEL and predicted:
            fp += 1
        else:
            tn += 1

    wall_seconds = time.perf_counter() - started
    metrics = binary_metrics(tp, fp, tn, fn)

    payload.update(
        {
            "metrics": metrics,
            "latency": {
                "load_ms": benchmark_common.latency_summary_ms(load_samples),
                "mfcc_ms": benchmark_common.latency_summary_ms(feature_samples),
                "predict_ms": benchmark_common.latency_summary_ms(predict_samples),
                "end_to_end_ms": benchmark_common.latency_summary_ms(total_samples),
            },
            "throughput": {
                "clips_per_second": _round_or_none(
                    benchmark_common.ratio(len(files), wall_seconds), 4
                ),
                "wall_seconds": round(wall_seconds, 4),
            },
            "diagnostics": {
                "mean_audio_duration_s": _round_or_none(
                    sum(durations) / len(durations) if durations else None, 4
                ),
                "min_confidence": _round_or_none(min(confidence_values) if confidence_values else None),
                "max_confidence": _round_or_none(max(confidence_values) if confidence_values else None),
                "mean_confidence": _round_or_none(
                    sum(confidence_values) / len(confidence_values) if confidence_values else None
                ),
                "label_threshold_mismatches": label_threshold_mismatches,
                "predictions": len(files),
            },
        }
    )
    return payload


def default_output_path() -> Path:
    stamp = benchmark_common.utc_timestamp()
    return ROOT / "benchmarks" / "results" / f"audio_resubstitution_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = base_payload(args)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    payload = evaluate(args, payload)
    output = Path(args.output) if args.output else default_output_path()
    payload["output"] = str(output)
    benchmark_common.write_json(payload, output)

    printable = {
        key: payload[key]
        for key in ("evaluation_kind", "dataset", "settings", "metrics", "latency", "throughput")
    }
    print(json.dumps(printable, indent=2))
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
