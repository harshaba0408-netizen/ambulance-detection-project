"""Reproducible YOLO evaluation on a named dataset split.

Usage:
    python scripts/evaluate_yolo.py --split val
    python scripts/evaluate_yolo.py --split test --output benchmarks/results/test.json
    python scripts/evaluate_yolo.py --dry-run

The script reports only what Ultralytics actually computes. It does not assume
a threshold from the dashboard: mAP evaluation uses `--conf 0.001` (the
Ultralytics default), because mAP is threshold-independent. Any operating-point
comparison must state the threshold explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ambulance_system.config import load_config  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_split_dir(data_yaml: Path, split: str) -> Path:
    import yaml

    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    entry = data.get(split)
    if entry is None and split == "valid":
        entry = data.get("val")
    if entry is None:
        raise ValueError(f"split {split!r} is not defined in {data_yaml}")
    return (data_yaml.parent / entry).resolve()


def dataset_counts(data_yaml: Path, split: str) -> dict[str, Any]:
    images_dir = _resolve_split_dir(data_yaml, split)
    labels_dir = images_dir.parent / "labels"

    images = (
        [path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        if images_dir.is_dir()
        else []
    )
    labels = list(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []

    return {
        "split": split,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "images": len(images),
        "labels": len(labels),
    }


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        info["ram_total_gb"] = round(memory.total / 1024**3, 2)
        info["ram_available_gb"] = round(memory.available / 1024**3, 2)
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = None
    return info


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.model.yolo_path, help="checkpoint path")
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "ambulance_images" / "data.yaml"),
        help="dataset YAML",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "valid", "test"),
        help="split to evaluate",
    )
    parser.add_argument("--imgsz", type=int, default=config.model.imgsz)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="confidence threshold for evaluation (mAP is threshold-independent)",
    )
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="disable Ultralytics plots (useful on low-memory CPU-only machines)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved settings and dataset counts without loading the model",
    )
    return parser


def base_payload(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    split = "valid" if args.split == "valid" else args.split
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_path}")

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "command": sys.argv,
        "environment": environment(),
        "model": {
            "path": str(model_path.resolve()),
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },
        "dataset": {
            "yaml": str(data_path.resolve()),
            **dataset_counts(data_path, split),
        },
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "eval_split": split,
        },
    }


def summarize_results(results: Any, wall_seconds: float) -> dict[str, Any]:
    """Extract metrics and speed from an Ultralytics validation result."""
    raw = dict(getattr(results, "results_dict", {}) or {})
    metrics = {
        "precision": _number(raw.get("metrics/precision(B)")),
        "recall": _number(raw.get("metrics/recall(B)")),
        "mAP50": _number(raw.get("metrics/mAP50(B)")),
        "mAP50-95": _number(raw.get("metrics/mAP50-95(B)")),
        "fitness": _number(raw.get("fitness")),
    }
    speed = {
        str(key): _number(value)
        for key, value in dict(getattr(results, "speed", {}) or {}).items()
    }
    return {
        "metrics": metrics,
        "speed_ms_per_image": speed,
        "wall_seconds": round(float(wall_seconds), 3),
        "ultralytics_save_dir": str(getattr(results, "save_dir", "")),
    }


def run_evaluation(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    split = payload["settings"]["eval_split"]
    output = Path(args.output) if args.output else None
    if output is None:
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "benchmarks" / "results" / f"yolo_eval_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    started = time.perf_counter()
    results = model.val(
        data=str(data_path),
        split=split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=not args.no_plots,
        project=str(output.parent),
        name=output.stem,
        exist_ok=True,
        verbose=False,
    )
    wall_seconds = time.perf_counter() - started

    payload.update(summarize_results(results, wall_seconds))
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output"] = str(output)
    return payload


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = base_payload(args)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    payload = run_evaluation(args, payload)
    print(json.dumps({key: payload[key] for key in ("model", "dataset", "settings", "metrics", "speed_ms_per_image")}, indent=2))
    print(f"written: {payload['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
