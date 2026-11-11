"""Tests for the Stage 10 benchmark harness.

These tests never load YOLO or the pickled classifier, and they do not require
the full local datasets. They cover the formatting helpers and the deterministic
system simulation only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_common  # noqa: E402


def _load(filename: str):
    name = f"stage10_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vision = _load("benchmark_vision.py")
audio = _load("benchmark_audio.py")
system = _load("benchmark_system.py")


# ---------------------------------------------------------------------------
# benchmark_common
# ---------------------------------------------------------------------------
def test_ratio_returns_none_for_zero_denominator():
    assert benchmark_common.ratio(0, 0) is None
    assert benchmark_common.ratio(3, 4) == 0.75


def test_percentile_uses_ceiling_index():
    values = [1.0, 2.0, 3.0, 4.0]
    assert benchmark_common.percentile(values, 0.0) == 1.0
    assert benchmark_common.percentile(values, 0.5) == 2.0
    assert benchmark_common.percentile(values, 0.95) == 4.0


def test_percentile_rejects_invalid_fraction():
    try:
        benchmark_common.percentile([1.0], 1.1)
    except ValueError as exc:
        assert "fraction" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_latency_summary_converts_seconds_to_milliseconds():
    summary = benchmark_common.latency_summary_ms([0.001, 0.002, 0.003])

    assert summary["count"] == 3
    assert summary["mean_ms"] == 2.0
    assert summary["median_ms"] == 2.0
    assert summary["p95_ms"] == 3.0
    assert summary["max_ms"] == 3.0


def test_latency_summary_of_empty_input_is_all_none():
    summary = benchmark_common.latency_summary_ms([])

    assert summary["count"] == 0
    assert summary["mean_ms"] is None
    assert summary["p95_ms"] is None


def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "sample.bin"
    target.write_bytes(b"ambulance")

    assert benchmark_common.sha256_file(target) == hashlib.sha256(b"ambulance").hexdigest()


def test_write_json_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "result.json"
    benchmark_common.write_json({"ok": True}, target)

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


# ---------------------------------------------------------------------------
# benchmark_vision
# ---------------------------------------------------------------------------
def test_pipeline_ms_sums_deployment_stages_and_excludes_loss():
    speed = {"preprocess": 1.0, "inference": 9.0, "postprocess": 1.0, "loss": 50.0}

    assert vision.pipeline_ms(speed) == 11.0


def test_pipeline_ms_returns_none_without_deployment_stages():
    assert vision.pipeline_ms({"loss": 2.0}) is None
    assert vision.pipeline_ms({}) is None


def test_derive_speed_reports_none_when_unmeasured():
    derived = vision.derive_speed({})

    assert derived["pipeline_ms_per_image"] is None
    assert derived["throughput_fps"] is None


def test_small_sample_caveat_marks_five_images():
    sample = vision.small_sample_caveat(5)

    assert sample["small_sample"] is True
    assert "5 labelled images" in sample["caveat"]
    assert vision.small_sample_caveat(100)["small_sample"] is False


# ---------------------------------------------------------------------------
# benchmark_audio
# ---------------------------------------------------------------------------
def test_binary_metrics_for_perfect_predictions():
    metrics = audio.binary_metrics(tp=200, fp=0, tn=200, fn=0)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["balanced_accuracy"] == 1.0


def test_binary_metrics_leave_undefined_ratios_as_none():
    metrics = audio.binary_metrics(tp=0, fp=0, tn=0, fn=0)

    assert metrics["accuracy"] is None
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["false_positive_rate"] is None
    assert metrics["balanced_accuracy"] is None


def test_content_duplicates_groups_identical_files(tmp_path):
    first = tmp_path / "a" / "one.wav"
    second = tmp_path / "b" / "two.wav"
    unique = tmp_path / "a" / "three.wav"
    for path, payload in ((first, b"same"), (second, b"same"), (unique, b"other")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    groups = audio.content_duplicates([first, second, unique])

    assert len(groups) == 1
    assert len(groups[0]["files"]) == 2
    assert groups[0]["classes"] == ["a", "b"]


# ---------------------------------------------------------------------------
# benchmark_system
# ---------------------------------------------------------------------------
def _view(name, phase, colour):
    return types.SimpleNamespace(name=name, phase=phase, colour=colour)


def test_changed_approaches_reports_only_changed_views():
    previous = {"Road 1": ("normal", "red")}
    snapshot = types.SimpleNamespace(
        approaches=[_view("Road 1", "normal", "red"), _view("Road 2", "normal", "red")]
    )

    assert system._changed_approaches(previous, snapshot) == []

    moved = {"Road 1": ("normal", "red")}
    snapshot = types.SimpleNamespace(
        approaches=[_view("Road 1", "emergency_priority", "green")]
    )
    assert system._changed_approaches(moved, snapshot) == [
        ("Road 1", "normal", "red", "emergency_priority/green")
    ]


def test_system_simulation_timeline_is_deterministic():
    from ambulance_system.config import load_config

    result = system.run_scenario(
        load_config(),
        approach="Road 2",
        with_evidence=True,
        step_s=0.5,
        duration_s=4.0,
        evidence_until_s=1.0,
        refresh_s=1.5,
        video_confidence=0.9,
        audio_confidence=0.92,
        resources=None,
    )

    timeline = result["simulated_timeline"]
    assert timeline["fusion_active_at_s"] == 0.5
    assert timeline["green_at_s"] == 2.0
    assert timeline["activation_delay_s"] == 2.0
    assert [item["to"] for item in timeline["signal_transitions"]] == [
        "emergency_priority/green"
    ]
    assert result["counters"]["signal_transitions"] == 1
    assert result["resources"]["before"] is None
