from __future__ import annotations

from pathlib import Path

import pytest

from ambulance_system.config import (
    AppConfig,
    ConfigError,
    DEFAULT_CONFIG_PATH,
    load_config,
)


def test_default_config_file_exists():
    assert DEFAULT_CONFIG_PATH.is_file()


def test_default_config_loads_and_resolves_model_paths():
    config = load_config()

    assert isinstance(config, AppConfig)
    assert Path(config.model.yolo_path).is_absolute()
    assert Path(config.model.yolo_path).is_file()
    assert Path(config.model.siren_classifier_path).is_file()
    assert config.model.confidence_threshold == pytest.approx(0.65)
    assert config.model.imgsz == 640


def test_default_sections_match_documented_values():
    config = load_config()

    assert config.video.source_type == "file"
    assert config.video.frame_skip == 5
    assert config.video.read_delay_s == pytest.approx(0.01)
    assert config.audio.sample_rate == 22050
    assert config.audio.window_s == pytest.approx(3.0)
    assert config.audio.overlap_s == pytest.approx(1.5)
    assert config.signal.release_threshold < config.signal.activation_threshold
    assert config.fusion.policy == "AUDIO"
    assert config.fusion.adaptive_window_s == pytest.approx(1.0)
    assert config.system.num_approaches == 4
    assert config.system.approach_names == ("Road 1", "Road 2", "Road 3", "Road 4")


def test_overrides_merge_without_dropping_siblings():
    config = load_config(overrides={"model": {"confidence_threshold": 0.8}})

    assert config.model.confidence_threshold == pytest.approx(0.8)
    assert config.model.imgsz == 640
    assert config.audio.window_s == pytest.approx(3.0)


def test_config_round_trips_through_dict():
    config = load_config()
    rebuilt = AppConfig.from_dict(config.to_dict())

    assert rebuilt == config


def test_path_resolution_uses_base_dir(tmp_path):
    config = load_config(base_dir=tmp_path)

    assert Path(config.model.yolo_path) == (tmp_path / "yolov8l_ambulance.pt").resolve()
    assert Path(config.model.siren_classifier_path) == (
        tmp_path / "siren_classifier.pkl"
    ).resolve()


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        AppConfig.from_dict({"model": {"confidance_threshold": 0.5}})


def test_confidence_outside_unit_interval_is_rejected():
    with pytest.raises(ConfigError, match="within"):
        AppConfig.from_dict({"model": {"confidence_threshold": 1.5}})
    with pytest.raises(ConfigError, match="within"):
        AppConfig.from_dict({"model": {"confidence_threshold": -0.1}})


def test_non_finite_confidence_is_rejected():
    with pytest.raises(ConfigError, match="finite"):
        AppConfig.from_dict({"model": {"confidence_threshold": float("nan")}})


def test_invalid_frame_skip_is_rejected():
    with pytest.raises(ConfigError, match="frame_skip"):
        AppConfig.from_dict({"video": {"frame_skip": 0}})


def test_negative_read_delay_is_rejected():
    with pytest.raises(ConfigError, match="read_delay_s"):
        AppConfig.from_dict({"video": {"read_delay_s": -0.1}})


def test_rtsp_requires_url():
    with pytest.raises(ConfigError, match="rtsp_url"):
        AppConfig.from_dict({"video": {"source_type": "rtsp"}})


def test_audio_overlap_must_be_smaller_than_window():
    with pytest.raises(ConfigError, match="overlap"):
        AppConfig.from_dict({"audio": {"overlap_s": 3.0, "window_s": 3.0}})


def test_release_threshold_cannot_exceed_activation_threshold():
    with pytest.raises(ConfigError, match="release_threshold"):
        AppConfig.from_dict(
            {"signal": {"activation_threshold": 0.5, "release_threshold": 0.9}}
        )


def test_invalid_fusion_policy_is_rejected():
    with pytest.raises(ConfigError, match="policy"):
        AppConfig.from_dict({"fusion": {"policy": "MAGIC"}})


def test_adaptive_window_must_be_positive_and_within_staleness():
    with pytest.raises(ConfigError, match="adaptive_window_s"):
        AppConfig.from_dict({"fusion": {"adaptive_window_s": 0}})
    with pytest.raises(ConfigError, match="adaptive_window_s"):
        AppConfig.from_dict(
            {"fusion": {"stale_after_s": 3.0, "adaptive_window_s": 4.0}}
        )


def test_invalid_log_level_is_rejected():
    with pytest.raises(ConfigError, match="logging.level"):
        AppConfig.from_dict({"logging": {"level": "LOUD"}})


def test_approach_name_count_must_match_num_approaches():
    with pytest.raises(ConfigError, match="approach_names"):
        AppConfig.from_dict({"system": {"num_approaches": 4, "approach_names": ["A", "B"]}})


def test_duplicate_approach_names_are_rejected():
    with pytest.raises(ConfigError, match="unique"):
        AppConfig.from_dict(
            {"system": {"num_approaches": 2, "approach_names": ["Road 1", "Road 1"]}}
        )


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_root_is_reported(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_config(bad)


def test_broken_yaml_is_reported(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("model: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)


# ---------------------------------------------------------------------------
# Type conversion and mapping validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "data, match",
    [
        ({"model": []}, "expected a mapping"),
        ({"model": {"yolo_path": 5}}, "expected string"),
        ({"model": {"imgsz": "640"}}, "expected int"),
        ({"model": {"imgsz": True}}, "expected int"),
        ({"model": {"confidence_threshold": "high"}}, "expected number"),
        ({"video": {"frame_skip": True}}, "expected int"),
        ({"video": {"reconnect_delay_s": "soon"}}, "expected number"),
        ({"video": {"source_type": 1}}, "expected string"),
        ({"system": {"approach_names": "Road 1"}}, "expected a sequence"),
        ({"system": {"approach_names": [1, 2]}}, "expected string"),
        ({"logging": {"level": "LOUD"}}, "logging.level"),
    ],
)
def test_type_errors_are_wrapped_as_config_errors(data, match):
    with pytest.raises(ConfigError, match=match):
        AppConfig.from_dict(data)


def test_empty_mapping_uses_defaults():
    assert AppConfig.from_dict({}) == AppConfig()


def test_integer_valued_float_is_accepted_for_int_fields():
    config = AppConfig.from_dict({"model": {"imgsz": 640.0}})

    assert config.model.imgsz == 640
    assert isinstance(config.model.imgsz, int)


def test_integer_is_accepted_for_float_fields():
    config = AppConfig.from_dict({"video": {"read_delay_s": 1}})

    assert config.video.read_delay_s == 1.0
    assert isinstance(config.video.read_delay_s, float)
