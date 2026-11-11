"""Measure one end-to-end video pipeline run.

This is a reproducibility check, not a benchmark suite. It reports only values
observed on the machine that ran it: delivered frames, drops, sampled
inferences and callback latency. It makes no real-time or traffic-control claim.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\benchmark_video_pipeline.py
    .\\venv\\Scripts\\python.exe scripts\\benchmark_video_pipeline.py --clip data\\Video\\2.mp4 --queue-size 8
    .\\venv\\Scripts\\python.exe scripts\\benchmark_video_pipeline.py --no-detection

The clip is decoded through `FileVideoSource`, so the numbers include the same
acquisition/drop behaviour the dashboard uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

from ambulance_system.config import load_config  # noqa: E402
from ambulance_system.detection import Detector  # noqa: E402
from ambulance_system.video import FileVideoSource, VideoPipeline  # noqa: E402


def clip_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "source_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "nominal_fps": round(float(capture.get(cv2.CAP_PROP_FPS)), 3),
        }
    finally:
        capture.release()


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip", default=str(ROOT / "data" / "Video" / "1.mp4"), help="video file to read"
    )
    parser.add_argument("--frame-skip", type=int, default=config.video.frame_skip)
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2,
        help="bounded frame queue; the dashboard uses 2",
    )
    parser.add_argument("--read-delay-s", type=float, default=config.video.read_delay_s)
    parser.add_argument("--imgsz", type=int, default=config.model.imgsz)
    parser.add_argument("--confidence", type=float, default=config.model.confidence_threshold)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="measure acquisition/delivery only; do not load YOLO",
    )
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    clip = Path(args.clip).expanduser().resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"clip not found: {clip}")

    detector: Optional[Detector] = None
    if not args.no_detection:
        config = load_config()
        detector = Detector(
            config.model.yolo_path,
            confidence_threshold=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
        )

    inferences = 0

    def callback(packet) -> None:
        nonlocal inferences
        if detector is not None and packet.sampled:
            inferences += 1
            detector.detect(packet.frame)

    pipeline = VideoPipeline(
        FileVideoSource(clip),
        callback,
        frame_skip=args.frame_skip,
        queue_size=args.queue_size,
        read_delay_s=args.read_delay_s,
    )
    started = time.perf_counter()
    pipeline.start()
    completed = pipeline.wait(timeout=args.timeout_s)
    wall_seconds = time.perf_counter() - started

    return {
        "clip": str(clip),
        "settings": {
            "frame_skip": args.frame_skip,
            "queue_size": args.queue_size,
            "read_delay_s": args.read_delay_s,
            "imgsz": None if detector is None else args.imgsz,
            "confidence": None if detector is None else args.confidence,
            "device": None if detector is None else args.device,
            "detection_enabled": detector is not None,
        },
        "clip_metadata": clip_metadata(clip),
        "wall_seconds": round(wall_seconds, 3),
        "completed": completed,
        "sampled_inferences": inferences,
        "stats": pipeline.stats.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"written: {output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
