from __future__ import annotations

import importlib.util
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_yolo.py"
MODEL_PATH = ROOT / "yolov8l_ambulance.pt"
DATA_YAML = ROOT / "data" / "ambulance_images" / "data.yaml"

_SPEC = importlib.util.spec_from_file_location("evaluate_yolo_script", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
evaluate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate)


class FakeResults:
    results_dict = {
        "metrics/precision(B)": 0.81,
        "metrics/recall(B)": 0.72,
        "metrics/mAP50(B)": 0.76,
        "metrics/mAP50-95(B)": 0.42,
        "fitness": 0.42,
    }
    speed = {"preprocess": 1.5, "inference": 120.0, "postprocess": 2.0}
    save_dir = Path("benchmarks/results/fake")


def test_dataset_counts_for_test_split_are_real():
    counts = evaluate.dataset_counts(DATA_YAML, "test")

    assert counts["images"] == 5
    assert counts["labels"] == 5
    assert counts["split"] == "test"


def test_dataset_counts_for_valid_split_are_real():
    counts = evaluate.dataset_counts(DATA_YAML, "valid")

    assert counts["images"] == 10
    assert counts["labels"] == 10


def test_summarize_results_extracts_metrics_and_speed():
    summary = evaluate.summarize_results(FakeResults(), 12.3456)

    assert summary["metrics"]["precision"] == 0.81
    assert summary["metrics"]["recall"] == 0.72
    assert summary["metrics"]["mAP50"] == 0.76
    assert summary["metrics"]["mAP50-95"] == 0.42
    assert summary["speed_ms_per_image"]["inference"] == 120.0
    assert summary["wall_seconds"] == 12.346
    assert summary["ultralytics_save_dir"].endswith("fake")


def test_missing_metric_becomes_none():
    class EmptyResults:
        results_dict = {}
        speed = {}
        save_dir = ""

    summary = evaluate.summarize_results(EmptyResults(), 0.0)

    assert summary["metrics"]["precision"] is None
    assert summary["speed_ms_per_image"] == {}


def test_base_payload_records_hashes_and_settings(tmp_path):
    args = types.SimpleNamespace(
        model=str(MODEL_PATH),
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.45,
        device="cpu",
    )

    payload = evaluate.base_payload(args)

    assert payload["model"]["sha256"].startswith("6784b16f")
    assert payload["model"]["size_bytes"] == 87657530
    assert payload["dataset"]["images"] == 5
    assert payload["settings"]["conf"] == 0.001
    assert payload["environment"]["python"]


def test_missing_model_is_reported(tmp_path):
    args = types.SimpleNamespace(
        model=str(tmp_path / "missing.pt"),
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.45,
        device="cpu",
    )

    try:
        evaluate.base_payload(args)
    except FileNotFoundError as exc:
        assert "model not found" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_parser_accepts_low_memory_flags():
    parser = evaluate.build_parser()
    args = parser.parse_args(["--split", "test", "--no-plots", "--imgsz", "320"])

    assert args.no_plots is True
    assert args.imgsz == 320
    assert args.split == "test"
