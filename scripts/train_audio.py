"""Train and evaluate a new siren classifier on a stored split.

The existing `siren_classifier.pkl` is never overwritten. A new Random Forest is
written under `artifacts/experiments/<timestamp>/`, together with its metrics,
split provenance and environment. The new model is an experiment only; wiring it
into the runtime requires a separate, explicit change.

The metrics label follows the split:

* `held_out_source_independent` when the split groups by verified session/source id.
* `held_out_content_group` when the split groups by acoustic content group. No
  content group spans splits, but content groups are not verified sources.
* `held_out_file_level` when the split only removes exact duplicates.

Only the first label may be described as source-independent.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\train_audio.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\train_audio.py
    .\\venv\\Scripts\\python.exe scripts\\train_audio.py --split data\\audio\\splits\\audio_split.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import benchmark_audio  # noqa: E402
import benchmark_common  # noqa: E402

DEFAULT_SPLIT = ROOT / "data" / "audio" / "splits" / "audio_split.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "experiments"
CLASS_LABEL = {"siren": 1, "non-siren": 0}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("train", "val", "test"):
        if key not in payload.get("splits", {}):
            raise ValueError(f"split file is missing splits.{key}")
    return payload


def evaluation_kind(split: dict[str, Any]) -> str:
    """Return the strongest honest label the split can support."""
    if split.get("source_independent"):
        return "held_out_source_independent"
    if split.get("grouping") == "content_group_id":
        return "held_out_content_group"
    return "held_out_file_level"


def feature_extraction(paths: list[str], *, sample_rate: int, duration_s: float, n_mfcc: int):
    """Load every path once and return (features, labels, timings)."""
    import numpy as np

    from ambulance_system.audio.features import load_audio, mfcc_features

    features: list[Any] = []
    labels: list[int] = []
    load_s: list[float] = []
    extract_s: list[float] = []
    for value in paths:
        path = resolve_path(value)
        class_name = path.parent.name
        if class_name not in CLASS_LABEL:
            raise ValueError(f"cannot infer class from {path}")
        started = time.perf_counter()
        samples, _ = load_audio(path, sample_rate=sample_rate, duration=duration_s)
        loaded = time.perf_counter()
        vector = mfcc_features(samples, sample_rate=sample_rate, n_mfcc=n_mfcc)
        finished = time.perf_counter()
        load_s.append(loaded - started)
        extract_s.append(finished - loaded)
        features.append(vector)
        labels.append(CLASS_LABEL[class_name])
    return np.asarray(features), np.asarray(labels), {
        "load_ms": benchmark_common.latency_summary_ms(load_s),
        "mfcc_ms": benchmark_common.latency_summary_ms(extract_s),
    }


def score(model, features, labels) -> dict[str, Any]:
    """Evaluate a fitted binary estimator and return honest metric tables."""
    import numpy as np

    if len(labels) == 0:
        return {"metrics": benchmark_audio.binary_metrics(0, 0, 0, 0), "samples": 0}
    predictions = np.asarray(model.predict(features)).reshape(-1).astype(int)
    truth = np.asarray(labels).reshape(-1).astype(int)
    tp = int(np.sum((truth == 1) & (predictions == 1)))
    fp = int(np.sum((truth == 0) & (predictions == 1)))
    tn = int(np.sum((truth == 0) & (predictions == 0)))
    fn = int(np.sum((truth == 1) & (predictions == 0)))
    return {
        "samples": int(len(truth)),
        "metrics": benchmark_audio.binary_metrics(tp, fp, tn, fn),
    }


def score_existing(features, labels, *, model_path: Path, sample_rate: int) -> dict[str, Any]:
    """Evaluate the shipped classifier on the same held-out vectors."""
    from ambulance_system.audio.classifier import SirenClassifier

    classifier = SirenClassifier.from_path(model_path, sample_rate=sample_rate)
    predictions = [1 if classifier.predict_features(vector).is_siren else 0 for vector in features]
    truth = [int(label) for label in labels]
    tp = sum(1 for t, p in zip(truth, predictions) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(truth, predictions) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(truth, predictions) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(truth, predictions) if t == 1 and p == 0)
    return {
        "path": str(model_path.resolve()),
        "sha256": benchmark_common.sha256_file(model_path),
        "samples": len(truth),
        "metrics": benchmark_audio.binary_metrics(tp, fp, tn, fn),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=str(DEFAULT_SPLIT), help="split JSON path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="experiment directory")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--existing-model",
        default=str(ROOT / "siren_classifier.pkl"),
        help="shipped classifier to evaluate on the same split (never modified)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"split file not found: {split_path}")
    split = load_split(split_path)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "split": str(split_path),
                    "source_independent": split.get("source_independent"),
                    "grouping": split.get("grouping"),
                    "evaluation_kind": evaluation_kind(split),
                    "counts": split.get("counts"),
                    "params": {
                        "n_estimators": args.n_estimators,
                        "seed": args.seed,
                        "sample_rate": args.sample_rate,
                        "duration_s": args.duration_s,
                        "n_mfcc": args.n_mfcc,
                    },
                },
                indent=2,
            )
        )
        return 0

    from sklearn.ensemble import RandomForestClassifier

    started = time.perf_counter()
    train_features, train_labels, train_timing = feature_extraction(
        split["splits"]["train"],
        sample_rate=args.sample_rate,
        duration_s=args.duration_s,
        n_mfcc=args.n_mfcc,
    )
    val_features, val_labels, val_timing = feature_extraction(
        split["splits"]["val"],
        sample_rate=args.sample_rate,
        duration_s=args.duration_s,
        n_mfcc=args.n_mfcc,
    )
    test_features, test_labels, test_timing = feature_extraction(
        split["splits"]["test"],
        sample_rate=args.sample_rate,
        duration_s=args.duration_s,
        n_mfcc=args.n_mfcc,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        max_features="sqrt",
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    fit_seconds = time.perf_counter() - started

    existing_path = Path(args.existing_model)
    kind = evaluation_kind(split)
    result = {
        "benchmark": "audio_training",
        "kind": "held_out_classifier_evaluation",
        "evaluation_kind": kind,
        "evaluation_validity": kind,
        "source_independent": bool(split.get("source_independent")),
        "duplicate_contaminated": kind == "held_out_file_level",
        "validity_note": (
            "Metrics are source-independent: groups are verified sessions/sources."
            if kind == "held_out_source_independent"
            else "Known acoustic content groups are held out, but content groups are "
            "not verified sessions or sources; these metrics are not source-independent."
            if kind == "held_out_content_group"
            else "File-level hold-out only; known duplicate content may span splits. "
            "These metrics are not source-independent."
        ),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "command": sys.argv,
        "environment": benchmark_common.environment(),
        "split": {
            "path": str(split_path.resolve()),
            "sha256": benchmark_common.sha256_file(split_path),
            "provenance_status": split.get("provenance_status"),
            "source_independent": bool(split.get("source_independent")),
            "grouping": split.get("grouping"),
            "evaluation_kind": kind,
            "counts": split.get("counts"),
            "caveat": split.get("caveat"),
        },
        "features": {
            "sample_rate": args.sample_rate,
            "duration_s": args.duration_s,
            "n_mfcc": args.n_mfcc,
        },
        "model": {
            "type": "RandomForestClassifier",
            "params": {
                "n_estimators": args.n_estimators,
                "random_state": args.seed,
                "max_features": "sqrt",
                "n_jobs": -1,
            },
            "training_samples": int(len(train_labels)),
            "fit_and_feature_seconds": round(fit_seconds, 3),
        },
        "data": {
            "train": {"samples": int(len(train_labels)), "timing": train_timing},
            "val": {"samples": int(len(val_labels)), "timing": val_timing},
            "test": {"samples": int(len(test_labels)), "timing": test_timing},
        },
        "metrics": {
            "val": score(model, val_features, val_labels),
            "test": score(model, test_features, test_labels),
        },
        "existing_model_on_same_split": {
            "val": score_existing(
                val_features, val_labels, model_path=existing_path, sample_rate=args.sample_rate
            ),
            "test": score_existing(
                test_features, test_labels, model_path=existing_path, sample_rate=args.sample_rate
            ),
        },
        "notes": (
            "The shipped siren_classifier.pkl is unchanged. This experiment is not "
            "wired into the runtime. Content-group metrics hold out acoustic groups, "
            "not verified sessions or sources. If source_independent is false, the "
            "scores are not source-independent and may overstate generalisation."
        ),
    }

    stamp = benchmark_common.utc_timestamp()
    output_dir = Path(args.output_dir) / f"audio_rf_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "siren_classifier.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    result["model"]["path"] = str(model_path.resolve())
    result["model"]["sha256"] = benchmark_common.sha256_file(model_path)
    result["model"]["size_bytes"] = model_path.stat().st_size
    result["output_dir"] = str(output_dir.resolve())
    benchmark_common.write_json(result, output_dir / "metrics.json")
    print(json.dumps({"evaluation_kind": result["evaluation_kind"], "metrics": result["metrics"]}, indent=2))
    print(f"written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
