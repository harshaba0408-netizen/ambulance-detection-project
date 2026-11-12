"""Stream live microphone audio through the siren classifier.

This is the Stage 6 live-input entry point. It captures mono blocks from an
input device, slides the same window used by the file pipeline, and prints one
line per classified window. Press Ctrl+C to stop.

Requirements and caveats:

* `sounddevice` is an optional runtime dependency (`pip install sounddevice`).
* The device is opened at the configured `audio.sample_rate`. PortAudio may
  reject a rate the device does not support; pick another device or rate.
* Two-channel devices are downmixed to mono by averaging.
* Nothing here is calibrated for a real intersection: the classifier was
  trained on short clips, and the result is "siren present in this window",
  not a validated live detection rate.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\stream_microphone.py --list-devices
    .\\venv\\Scripts\\python.exe scripts\\stream_microphone.py
    .\\venv\\Scripts\\python.exe scripts\\stream_microphone.py --device 1 --seconds 20
    .\\venv\\Scripts\\python.exe scripts\\stream_microphone.py --threshold 0.6
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ambulance_system.audio import (  # noqa: E402
    AudioSourceError,
    MicrophoneAudioSource,
    SirenClassifier,
    list_input_devices,
    stream_predictions,
)
from ambulance_system.config import AppConfig, ConfigError, load_config  # noqa: E402


def parse_device(value: str) -> str | int:
    """Accept an index (`1`) or a name fragment (`Microphone`)."""
    text = value.strip()
    if text.isdigit():
        return int(text)
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list input devices and exit; no capture is started",
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default=None,
        help="input device index or name fragment (default: audio.device, else system default)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="stop after N seconds, checked after each completed window (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the classifier siren threshold from the config",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="classifier pickle (default: model.siren_classifier_path)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: config/default.yaml)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        default=None,
        help="capture channels before mono downmix (default: audio.channels)",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=1e-3,
        help="live silence gate: windows below this RMS are reported without "
        "classification (default: 0.001; use 0 to disable)",
    )
    return parser


def _load(args: argparse.Namespace) -> AppConfig:
    try:
        return load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        try:
            devices = list_input_devices()
        except AudioSourceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not devices:
            print("no input devices reported by the audio backend")
            return 0
        for device in devices:
            print(
                f"[{device.index}] {device.name} "
                f"({device.max_input_channels} in, {device.default_samplerate:.0f} Hz)"
            )
        return 0

    if args.seconds is not None and args.seconds <= 0:
        print("error: --seconds must be > 0", file=sys.stderr)
        return 2
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        print("error: --threshold must be within [0, 1]", file=sys.stderr)
        return 2
    if not math.isfinite(args.min_rms) or args.min_rms < 0.0:
        print("error: --min-rms must be a finite number >= 0", file=sys.stderr)
        return 2

    config = _load(args)
    audio = config.audio
    threshold = audio.threshold if args.threshold is None else float(args.threshold)
    model_path = args.model or Path(config.model.siren_classifier_path)
    try:
        classifier = SirenClassifier.from_path(
            model_path,
            threshold=threshold,
            sample_rate=audio.sample_rate,
        )
    except (OSError, ValueError) as exc:
        print(f"error: could not load classifier: {exc}", file=sys.stderr)
        return 2

    # The first MFCC call can take seconds (library warm-up). Do it before the
    # device is opened, otherwise the live queue overflows and blocks are lost.
    warmup = np.zeros(max(1, int(round(audio.window_s * audio.sample_rate))), dtype=np.float32)
    try:
        classifier.predict_samples(warmup)
    except Exception as exc:  # pragma: no cover - model dependent
        print(f"error: classifier warm-up failed: {exc}", file=sys.stderr)
        return 2

    source = MicrophoneAudioSource(
        device=args.device if args.device is not None else audio.device,
        sample_rate=audio.sample_rate,
        block_s=audio.block_s,
        channels=args.channels or audio.channels,
    )

    started = time.monotonic()
    windows = 0
    sirens = 0
    silent = 0
    predictions = None
    gate = (
        f"silence gate RMS < {args.min_rms:g}"
        if args.min_rms > 0.0
        else "silence gate off"
    )
    print(
        f"capturing from {source.name} at {source.sample_rate} Hz "
        f"(window {audio.window_s:.1f}s, overlap {audio.overlap_s:.1f}s, "
        f"threshold {threshold:.2f}, {gate}); Ctrl+C to stop",
        flush=True,
    )
    try:
        predictions = stream_predictions(
            source,
            classifier,
            window_s=audio.window_s,
            overlap_s=audio.overlap_s,
            min_rms=args.min_rms,
        )
        for prediction in predictions:
            windows += 1
            if prediction.is_siren:
                sirens += 1
            if prediction.confidence is None and not prediction.is_siren:
                silent += 1
                label = "silent"
            else:
                label = "SIREN" if prediction.is_siren else "quiet"
            confidence = (
                "n/a" if prediction.confidence is None else f"{prediction.confidence:.3f}"
            )
            print(
                f"{prediction.window.start_s:8.2f}s  {label:6s}  conf={confidence}",
                flush=True,
            )
            if args.seconds is not None and time.monotonic() - started >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
    except AudioSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            if predictions is not None:
                predictions.close()
        except Exception:
            pass
        source.close()

    if source.dropped_blocks:
        print(
            f"warning: dropped {source.dropped_blocks} block(s) because the "
            "consumer fell behind",
            file=sys.stderr,
        )
    if source.invalid_blocks:
        print(
            f"warning: rejected {source.invalid_blocks} invalid block(s) from "
            f"{source.name} (empty, non-finite or outside +/-4.0); check the "
            "microphone privacy setting and device driver",
            file=sys.stderr,
        )
    summary = f"summary: {windows} windows, {sirens} siren window(s)"
    if silent:
        summary += f", {silent} below the silence gate"
    print(summary)
    if windows == 0:
        print("error: no complete window was captured", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
