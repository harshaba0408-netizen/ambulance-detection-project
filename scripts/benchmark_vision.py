"""Measure YOLO detection quality and speed on one dataset split.

Every reported value comes from an Ultralytics `val()` run on this machine. The
script stores the model hash, dataset counts, environment string and the exact
command in the JSON result so the number can be traced and repeated.

This is a measurement harness, not a training or tuning tool. The checkpoint is
never modified, and no value is estimated from model metadata.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\benchmark_vision.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\benchmark_vision.py --split test --imgsz 640 --no-plots
    .\\venv\\Scripts\\python.exe scripts\\benchmark_vision.py --split test --output benchmarks\\results\\vision_test.json

mAP is threshold-independent; it uses Ultralytics' default `conf=0.001`. Any
operating-point precision/recall comparison must state its threshold, which is
recorded under `settings.conf`.
"""

from __future__ import annotations

import argparse
import json
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

from ambulance_system.config import load_config  # noqa: E402

import benchmark_common  # noqa: E402
import evaluate_yolo  # noqa: E402

# A split this small cannot support a statistical claim about real traffic.
SMALL_SAMPLE_IMAGES = 30


def pipeline_ms(speed: dict[str, Any]) -> Optional[float]:
    """Sum the Ultralytics per-image stages that are part of one inference.

    `loss` is not part of deployment latency and is deliberately excluded.
    Absent stages are omitted from the sum; if no deployment stage is present
    the result is `None`.
    """
    stages = ("preprocess", "inference", "postprocess")
    present = [float(speed[key]) for key in stages if speed.get(key) is not None]
    if not present:
        return None
    return float(sum(present))


def fps_from_ms(milliseconds: Optional[float]) -> Optional[float]:
    if milliseconds is None or milliseconds <= 0.0:
        return None
    return 1000.0 / float(milliseconds)


def derive_speed(speed: dict[str, Any]) -> dict[str, Any]:
    """Turn Ultralytics stage timings into deployment-oriented values."""
    total = pipeline_ms(speed)
    return {
        "pipeline_ms_per_image": None if total is None else round(total, 4),
        "throughput_fps": None if total is None else round(fps_from_ms(total), 4),
        "formula": "1000 / (preprocess_ms + inference_ms + postprocess_ms)",
        "note": "Validation-loop throughput on CPU; not a live-camera guarantee.",
    }


def small_sample_caveat(images: int) -> dict[str, Any]:
    flagged = images < SMALL_SAMPLE_IMAGES
    return {
        "images": images,
        "threshold": SMALL_SAMPLE_IMAGES,
        "small_sample": flagged,
        "caveat": (
            f"Only {images} labelled images; metrics are indicative, not a statistical claim."
            if flagged
            else None
        ),
    }


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
        default="test",
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

    counts = evaluate_yolo.dataset_counts(data_path, split)
    return {
        "benchmark": "vision",
        "kind": "measured_model_evaluation",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "command": sys.argv,
        "environment": benchmark_common.environment(),
        "model": {
            "path": str(model_path.resolve()),
            "size_bytes": model_path.stat().st_size,
            "sha256": benchmark_common.sha256_file(model_path),
            "training_unchanged": True,
        },
        "dataset": {
            "yaml": str(data_path.resolve()),
            **counts,
            "sample": small_sample_caveat(int(counts["images"])),
        },
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "eval_split": split,
            "plots": not args.no_plots,
        },
    }


def run_evaluation(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    split = payload["settings"]["eval_split"]
    output = Path(args.output) if args.output else default_output_path(split, args.imgsz)
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

    summary = evaluate_yolo.summarize_results(results, wall_seconds)
    payload.update(summary)
    payload["derived_speed"] = derive_speed(summary.get("speed_ms_per_image", {}))
    payload["output"] = str(output)
    benchmark_common.write_json(payload, output)
    return payload


def default_output_path(split: str, imgsz: int) -> Path:
    stamp = benchmark_common.utc_timestamp()
    return ROOT / "benchmarks" / "results" / f"vision_{split}_{imgsz}_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = base_payload(args)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    payload = run_evaluation(args, payload)
    printable = {
        key: payload[key]
        for key in ("model", "dataset", "settings", "metrics", "speed_ms_per_image", "derived_speed")
    }
    print(json.dumps(printable, indent=2))
    print(f"written: {payload['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
