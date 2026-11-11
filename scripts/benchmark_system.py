"""Measure the dashboard's software decision path with synthetic evidence.

No camera, microphone, YOLO checkpoint or audio classifier is loaded. The
scenario feeds fixed-confidence synthetic events into `DashboardViewModel` and
uses `FakeClock` for simulated seconds. The output therefore separates two kinds
of number:

* `wall_latency`: real `perf_counter` timings of Python calls on this machine;
* `simulated_timeline`: timestamps produced by `FakeClock`, not wall time.

This measures the decision path only. It makes no claim about real-time video
inference, detection accuracy or traffic-control performance.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\benchmark_system.py --dry-run
    .\\venv\\Scripts\\python.exe scripts\\benchmark_system.py
    .\\venv\\Scripts\\python.exe scripts\\benchmark_system.py --output benchmarks\\results\\system.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from ambulance_system.clock import FakeClock  # noqa: E402
from ambulance_system.config import AppConfig, load_config  # noqa: E402
from ambulance_system.dashboard.viewmodel import DashboardSnapshot, DashboardViewModel  # noqa: E402
from ambulance_system.monitoring.metrics import ResourceMonitor  # noqa: E402

import benchmark_common  # noqa: E402


def _state(snapshot: DashboardSnapshot) -> dict[str, tuple[str, str]]:
    return {
        view.name: (view.phase, view.colour)
        for view in snapshot.approaches
    }


def _changed_approaches(
    previous: dict[str, tuple[str, str]], snapshot: DashboardSnapshot
) -> list[tuple[str, str, str, str]]:
    changes: list[tuple[str, str, str, str]] = []
    for view in snapshot.approaches:
        old = previous.get(view.name)
        if old is None:
            continue
        new = (view.phase, view.colour)
        if old != new:
            changes.append((view.name, old[0], old[1], f"{view.phase}/{view.colour}"))
    return changes


def run_scenario(
    config: AppConfig,
    *,
    approach: str,
    with_evidence: bool,
    step_s: float,
    duration_s: float,
    evidence_until_s: float,
    refresh_s: float,
    video_confidence: float,
    audio_confidence: float,
    resources: Optional[ResourceMonitor] = None,
) -> dict[str, Any]:
    clock = FakeClock(start=0.0)
    viewmodel = DashboardViewModel(config, clock=clock)

    resource_before = resources.sample().to_dict() if resources is not None else None

    started = time.perf_counter()
    start_call_s = _timed(viewmodel.start)
    observe_video_s: list[float] = []
    observe_audio_s: list[float] = []
    tick_s: list[float] = []
    transition_tick_s: list[float] = []

    previous = _state(viewmodel.snapshot())
    first_fused_active: Optional[float] = None
    first_green: Optional[float] = None
    first_yellow: Optional[float] = None
    first_red_after_green: Optional[float] = None
    first_normal_after_green: Optional[float] = None
    transitions: list[dict[str, Any]] = []

    if with_evidence:
        for observe, confidence, kind in (
            (viewmodel.observe_video, video_confidence, "video"),
            (viewmodel.observe_audio, audio_confidence, "audio"),
        ):
            duration = _timed_observe(
                observe, approach, confidence, kind, timestamp=clock.now()
            )
            (observe_video_s if kind == "video" else observe_audio_s).append(duration)

    next_refresh = refresh_s
    steps = int(round(duration_s / step_s))
    for _ in range(steps):
        clock.advance(step_s)
        simulated = clock.now()

        if with_evidence and simulated <= evidence_until_s + 1e-9 and simulated + 1e-9 >= next_refresh:
            observe_video_s.append(
                _timed_observe(
                    viewmodel.observe_video,
                    approach,
                    video_confidence,
                    "video",
                    timestamp=simulated,
                )
            )
            observe_audio_s.append(
                _timed_observe(
                    viewmodel.observe_audio,
                    approach,
                    audio_confidence,
                    "audio",
                    timestamp=simulated,
                )
            )
            next_refresh += refresh_s

        tick_started = time.perf_counter()
        snapshot = viewmodel.tick()
        tick_elapsed = time.perf_counter() - tick_started
        tick_s.append(tick_elapsed)

        view = snapshot.approach(approach)
        if first_fused_active is None and view.fused_active:
            first_fused_active = simulated
        if first_green is None and view.phase == "emergency_priority":
            first_green = simulated
        if first_green is not None and first_yellow is None and view.phase == "clearing":
            first_yellow = simulated
        if first_yellow is not None and first_red_after_green is None and view.colour == "red":
            first_red_after_green = simulated
        if first_red_after_green is not None and first_normal_after_green is None and view.phase == "normal":
            first_normal_after_green = simulated

        changes = _changed_approaches(previous, snapshot)
        if changes:
            transition_tick_s.append(tick_elapsed)
            for name, old_phase, old_colour, new_state in changes:
                transitions.append(
                    {
                        "simulated_time_s": round(simulated, 6),
                        "approach": name,
                        "from": f"{old_phase}/{old_colour}",
                        "to": new_state,
                    }
                )
            previous = _state(snapshot)

    wall_seconds = time.perf_counter() - started
    final_snapshot = viewmodel.snapshot()
    snapshot_s = [_timed(viewmodel.snapshot) for _ in range(200)]

    resource_after = resources.sample().to_dict() if resources is not None else None

    activation_delay = (
        None if first_green is None else round(first_green - 0.0, 6)
    )
    green_duration = (
        None
        if first_green is None or first_yellow is None
        else round(first_yellow - first_green, 6)
    )
    clearance_duration = (
        None
        if first_yellow is None or first_normal_after_green is None
        else round(first_normal_after_green - first_yellow, 6)
    )
    first_fused_active = _round6(first_fused_active)
    first_green = _round6(first_green)
    first_yellow = _round6(first_yellow)
    first_red_after_green = _round6(first_red_after_green)
    first_normal_after_green = _round6(first_normal_after_green)

    return {
        "name": "emergency" if with_evidence else "idle",
        "evidence_enabled": with_evidence,
        "wall_latency": {
            "start_ms": _ms(start_call_s),
            "observe_video_ms": benchmark_common.latency_summary_ms(observe_video_s),
            "observe_audio_ms": benchmark_common.latency_summary_ms(observe_audio_s),
            "tick_ms": benchmark_common.latency_summary_ms(tick_s),
            "transition_tick_ms": benchmark_common.latency_summary_ms(transition_tick_s),
            "snapshot_ms": benchmark_common.latency_summary_ms(snapshot_s),
        },
        "simulated_timeline": {
            "evidence_at_s": 0.0 if with_evidence else None,
            "fusion_active_at_s": first_fused_active,
            "green_at_s": first_green,
            "activation_delay_s": activation_delay,
            "yellow_at_s": first_yellow,
            "green_duration_s": green_duration,
            "red_at_s": first_red_after_green,
            "normal_at_s": first_normal_after_green,
            "clearance_duration_s": clearance_duration,
            "signal_transitions": transitions,
            "time_is_simulated": True,
        },
        "counters": {
            "ticks": len(tick_s),
            "evidence_submissions": len(observe_video_s) + len(observe_audio_s),
            "signal_transitions": len(transitions),
            "event_log_entries": len(final_snapshot.events),
            "final_status": final_snapshot.status_message,
            "final_fault": final_snapshot.fault,
        },
        "resources": {
            "before": resource_before,
            "after": resource_after,
            "process_ram_delta_mb": _delta_mb(resource_before, resource_after),
        },
        "wall_seconds": round(wall_seconds, 6),
    }


def _timed(callable_obj):
    started = time.perf_counter()
    callable_obj()
    return time.perf_counter() - started


def _timed_observe(observe, approach, confidence, kind, *, timestamp: float) -> float:
    started = time.perf_counter()
    if kind == "video":
        observe(approach, confidence, detected=True, timestamp=timestamp)
    else:
        observe(approach, confidence, is_siren=True, timestamp=timestamp)
    return time.perf_counter() - started


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 6)


def _round6(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 6)


def _delta_mb(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> Optional[float]:
    if not before or not after:
        return None
    first = before.get("process_ram_mb")
    second = after.get("process_ram_mb")
    if first is None or second is None:
        return None
    return round(float(second) - float(first), 4)


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approach",
        default="Road 2" if "Road 2" in config.system.approach_names else config.system.approach_names[0],
        choices=config.system.approach_names,
        help="approach that receives synthetic emergency evidence",
    )
    parser.add_argument("--step-s", type=float, default=0.1, help="simulated decision interval")
    parser.add_argument("--duration-s", type=float, default=40.0, help="simulated run length")
    parser.add_argument(
        "--evidence-until-s",
        type=float,
        default=12.0,
        help="stop refreshing evidence after this simulated time",
    )
    parser.add_argument(
        "--refresh-s",
        type=float,
        default=1.5,
        help="evidence refresh interval; must exceed fusion debounce_s",
    )
    parser.add_argument("--video-confidence", type=float, default=0.90)
    parser.add_argument("--audio-confidence", type=float, default=0.92)
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved scenario and configuration without running it",
    )
    return parser


def base_payload(args: argparse.Namespace, config: AppConfig) -> dict[str, Any]:
    return {
        "benchmark": "system",
        "kind": "software_decision_path_simulation",
        "generated_at": benchmark_common.iso_now(),
        "command": sys.argv,
        "environment": benchmark_common.environment(),
        "model_inference_included": False,
        "camera_or_microphone_included": False,
        "caveat": (
            "Synthetic evidence is fed directly into the dashboard view-model. "
            "Wall timings are Python decision-path costs. Simulated seconds come "
            "from FakeClock and are not wall-clock measurements of a live system."
        ),
        "resource_note": (
            "CPU/RAM are single psutil samples. cpu_percent(interval=None) compares "
            "against the previous call, so the first sample can be 0.0 or stale; "
            "process_ram_mb is resident memory and includes interpreter imports."
        ),
        "scenario": {
            "approach": args.approach,
            "step_s": args.step_s,
            "duration_s": args.duration_s,
            "evidence_until_s": args.evidence_until_s,
            "refresh_s": args.refresh_s,
            "video_confidence": args.video_confidence,
            "audio_confidence": args.audio_confidence,
            "signal": asdict(config.signal),
            "fusion": asdict(config.fusion),
            "approaches": list(config.system.approach_names),
        },
    }


def default_output_path() -> Path:
    stamp = benchmark_common.utc_timestamp()
    return ROOT / "benchmarks" / "results" / f"system_simulation_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    if args.step_s <= 0 or args.duration_s <= 0:
        raise ValueError("step_s and duration_s must be positive")
    if args.refresh_s <= config.fusion.debounce_s:
        raise ValueError("refresh_s must exceed fusion.debounce_s to avoid duplicate suppression")
    if args.evidence_until_s > args.duration_s:
        raise ValueError("evidence_until_s must not exceed duration_s")

    payload = base_payload(args, config)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    resources = ResourceMonitor()
    payload["scenarios"] = [
        run_scenario(
            config,
            approach=args.approach,
            with_evidence=False,
            step_s=args.step_s,
            duration_s=args.duration_s,
            evidence_until_s=args.evidence_until_s,
            refresh_s=args.refresh_s,
            video_confidence=args.video_confidence,
            audio_confidence=args.audio_confidence,
            resources=resources,
        ),
        run_scenario(
            config,
            approach=args.approach,
            with_evidence=True,
            step_s=args.step_s,
            duration_s=args.duration_s,
            evidence_until_s=args.evidence_until_s,
            refresh_s=args.refresh_s,
            video_confidence=args.video_confidence,
            audio_confidence=args.audio_confidence,
            resources=resources,
        ),
    ]

    output = Path(args.output) if args.output else default_output_path()
    payload["output"] = str(output)
    benchmark_common.write_json(payload, output)
    print(json.dumps({"scenarios": payload["scenarios"]}, indent=2))
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
