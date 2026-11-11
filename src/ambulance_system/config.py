"""Typed application configuration.

`config/default.yaml` is the single source of truth for runtime settings.
Values are validated when the config object is built, so a typo fails fast at
start-up instead of surfacing as a silent default deep inside the pipeline.

No secrets belong in this file or in these dataclasses. Credentials for future
camera or broker integrations must come from the environment, never from YAML
that is committed to Git.
"""

from __future__ import annotations

import math
import types
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Union, get_args, get_origin, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_VIDEO_SOURCES = {"file", "webcam", "rtsp"}
VALID_POLICIES = {"VIDEO", "AUDIO", "BOTH", "ADAPTIVE"}


class ConfigError(ValueError):
    """Raised when a configuration file is structurally or semantically invalid."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _unit_interval(value: float, name: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"{name} must be a finite number",
    )
    _require(0.0 <= float(value) <= 1.0, f"{name} must be within [0.0, 1.0], got {value!r}")


def _non_negative(value: float, name: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"{name} must be a finite number",
    )
    _require(float(value) >= 0.0, f"{name} must be >= 0, got {value!r}")


def _positive(value: float, name: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"{name} must be a finite number",
    )
    _require(float(value) > 0.0, f"{name} must be > 0, got {value!r}")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Paths and thresholds for the two baseline models."""

    yolo_path: str = "yolov8l_ambulance.pt"
    siren_classifier_path: str = "siren_classifier.pkl"
    confidence_threshold: float = 0.65
    imgsz: int = 640

    def __post_init__(self) -> None:
        _require(bool(str(self.yolo_path).strip()), "model.yolo_path must not be empty")
        _require(
            bool(str(self.siren_classifier_path).strip()),
            "model.siren_classifier_path must not be empty",
        )
        _unit_interval(self.confidence_threshold, "model.confidence_threshold")
        _require(self.imgsz > 0, "model.imgsz must be > 0")
        _require(self.imgsz % 32 == 0, "model.imgsz must be a multiple of 32")


@dataclass(frozen=True)
class VideoConfig:
    """Video acquisition settings shared by file, webcam and RTSP sources."""

    source_type: str = "file"
    file_path: Optional[str] = None
    webcam_index: int = 0
    rtsp_url: Optional[str] = None
    frame_skip: int = 5
    read_delay_s: float = 0.01
    reconnect_delay_s: float = 2.0
    read_timeout_s: float = 5.0
    display_width: int = 320
    display_height: int = 180

    def __post_init__(self) -> None:
        _require(
            self.source_type in VALID_VIDEO_SOURCES,
            f"video.source_type must be one of {sorted(VALID_VIDEO_SOURCES)}, "
            f"got {self.source_type!r}",
        )
        _require(self.frame_skip >= 1, "video.frame_skip must be >= 1")
        _require(self.webcam_index >= 0, "video.webcam_index must be >= 0")
        _non_negative(self.read_delay_s, "video.read_delay_s")
        _non_negative(self.reconnect_delay_s, "video.reconnect_delay_s")
        _positive(self.read_timeout_s, "video.read_timeout_s")
        _require(self.display_width > 0, "video.display_width must be > 0")
        _require(self.display_height > 0, "video.display_height must be > 0")
        if self.source_type == "rtsp":
            _require(
                bool(str(self.rtsp_url or "").strip()),
                "video.rtsp_url is required when source_type is 'rtsp'",
            )


@dataclass(frozen=True)
class AudioConfig:
    """Streaming audio window settings and classifier threshold."""

    sample_rate: int = 22050
    window_s: float = 3.0
    overlap_s: float = 1.5
    block_s: float = 0.25
    threshold: float = 0.5
    channels: int = 1
    device: Optional[str] = None

    def __post_init__(self) -> None:
        _require(self.sample_rate > 0, "audio.sample_rate must be > 0")
        _positive(self.window_s, "audio.window_s")
        _non_negative(self.overlap_s, "audio.overlap_s")
        _require(
            self.overlap_s < self.window_s,
            "audio.overlap_s must be smaller than audio.window_s",
        )
        _positive(self.block_s, "audio.block_s")
        _require(
            self.block_s <= self.window_s,
            "audio.block_s must not exceed audio.window_s",
        )
        _unit_interval(self.threshold, "audio.threshold")
        _require(self.channels in (1, 2), "audio.channels must be 1 or 2")


@dataclass(frozen=True)
class SignalConfig:
    """Timing and hysteresis for the simulated signal state machine."""

    activation_threshold: float = 0.65
    release_threshold: float = 0.50
    min_confirmation_s: float = 1.0
    release_grace_s: float = 2.0
    max_priority_s: float = 30.0
    min_green_s: float = 5.0
    yellow_s: float = 3.0
    all_red_s: float = 2.0
    fail_safe_state: str = "ALL_RED"

    def __post_init__(self) -> None:
        _unit_interval(self.activation_threshold, "signal.activation_threshold")
        _unit_interval(self.release_threshold, "signal.release_threshold")
        _require(
            self.release_threshold <= self.activation_threshold,
            "signal.release_threshold must be <= signal.activation_threshold",
        )
        _non_negative(self.min_confirmation_s, "signal.min_confirmation_s")
        _non_negative(self.release_grace_s, "signal.release_grace_s")
        _positive(self.max_priority_s, "signal.max_priority_s")
        _positive(self.min_green_s, "signal.min_green_s")
        _positive(self.yellow_s, "signal.yellow_s")
        _positive(self.all_red_s, "signal.all_red_s")
        _require(
            self.max_priority_s >= self.min_green_s,
            "signal.max_priority_s must be >= signal.min_green_s",
        )
        _require(
            self.fail_safe_state == "ALL_RED",
            "signal.fail_safe_state must be 'ALL_RED' in the software simulator",
        )


@dataclass(frozen=True)
class FusionConfig:
    """Evidence policy and event lifetime settings."""

    policy: str = "AUDIO"
    stale_after_s: float = 3.0
    debounce_s: float = 1.0
    adaptive_window_s: float = 1.0

    def __post_init__(self) -> None:
        _require(
            self.policy in VALID_POLICIES,
            f"fusion.policy must be one of {sorted(VALID_POLICIES)}, got {self.policy!r}",
        )
        _positive(self.stale_after_s, "fusion.stale_after_s")
        _non_negative(self.debounce_s, "fusion.debounce_s")
        _positive(self.adaptive_window_s, "fusion.adaptive_window_s")
        _require(
            self.adaptive_window_s <= self.stale_after_s,
            "fusion.adaptive_window_s must be <= fusion.stale_after_s",
        )


@dataclass(frozen=True)
class LoggingConfig:
    """Structured logging destinations. `None` disables that destination."""

    level: str = "INFO"
    text_file: Optional[str] = None
    json_file: Optional[str] = None

    def __post_init__(self) -> None:
        level = str(self.level).upper()
        _require(level in VALID_LOG_LEVELS, f"logging.level must be one of {sorted(VALID_LOG_LEVELS)}")
        object.__setattr__(self, "level", level)


@dataclass(frozen=True)
class MonitoringConfig:
    """Sampling cadence for FPS, latency and resource metrics."""

    metrics_interval_s: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.metrics_interval_s, "monitoring.metrics_interval_s")


@dataclass(frozen=True)
class SystemConfig:
    """Top-level system shape."""

    num_approaches: int = 4
    approach_names: tuple[str, ...] = ("Road 1", "Road 2", "Road 3", "Road 4")

    def __post_init__(self) -> None:
        _require(1 <= self.num_approaches <= 8, "system.num_approaches must be within [1, 8]")
        _require(
            len(self.approach_names) == self.num_approaches,
            "system.approach_names must contain exactly system.num_approaches entries",
        )
        _require(len(set(self.approach_names)) == len(self.approach_names), "approach names must be unique")
        _require(all(str(name).strip() for name in self.approach_names), "approach names must not be empty")


@dataclass(frozen=True)
class AppConfig:
    """Validated configuration tree for the whole application."""

    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_paths(self, root: str | Path) -> "AppConfig":
        """Return a copy whose model paths are absolute, relative to `root`."""
        root_path = Path(root)

        def _absolute(value: str) -> str:
            path = Path(value).expanduser()
            return str(path if path.is_absolute() else (root_path / path).resolve())

        return replace(
            self,
            model=replace(
                self.model,
                yolo_path=_absolute(self.model.yolo_path),
                siren_classifier_path=_absolute(self.model.siren_classifier_path),
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        return _build(cls, data, "config")


# ---------------------------------------------------------------------------
# Dataclass construction from plain mappings
# ---------------------------------------------------------------------------
def _convert(value: Any, expected: Any, path: str) -> Any:
    if expected is Any:
        return value

    if is_dataclass(expected) and isinstance(expected, type):
        return _build(expected, value, path)

    origin = get_origin(expected)

    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(expected) if arg is not type(None)]
        if value is None:
            return None
        if len(args) != 1:
            raise ConfigError(f"{path}: unsupported union type {expected!r}")
        return _convert(value, args[0], path)

    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value

    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value

    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected number, got {value!r}")
        return float(value)

    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected string, got {value!r}")
        return value

    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a sequence, got {value!r}")
        args = get_args(expected)
        item_type = args[0] if args else Any
        return tuple(_convert(item, item_type, f"{path}[{index}]") for index, item in enumerate(value))

    raise ConfigError(f"{path}: unsupported configuration type {expected!r}")


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {', '.join(unknown)}")

    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        kwargs[item.name] = _convert(data[item.name], hints[item.name], f"{path}.{item.name}")

    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path | None = None,
    *,
    base_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load, validate and path-resolve a YAML configuration file.

    Relative model paths are resolved against `base_dir` when supplied, or
    against the repository root that contains the config file.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path}: root must be a mapping")

    raw = dict(raw)
    if overrides:
        raw = _deep_merge(raw, overrides)

    config = AppConfig.from_dict(raw)
    root = Path(base_dir) if base_dir is not None else config_path.resolve().parent.parent
    return config.resolve_paths(root)
