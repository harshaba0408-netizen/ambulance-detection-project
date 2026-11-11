# Configuration and Event Schema

Stage 2 introduced a typed configuration layer and a shared event schema. Since
Stage 4, `new.py` loads `config/default.yaml` at start-up and uses it for
model paths and thresholds; Stage 5 added the video pipeline settings, Stage 6
the audio streaming settings, and Stage 8 the dashboard's per-approach policy
defaults, health cadence and display size. The event schema from Stage 2 is the
evidence unit consumed by fusion in Stage 7.

## 1. Loading configuration

```python
from ambulance_system.config import load_config

config = load_config()                       # config/default.yaml
config = load_config("config/custom.yaml")   # explicit file
config = load_config(overrides={"model": {"confidence_threshold": 0.8}})
```

Behaviour:

* Relative model paths are resolved against the repository root, or against
  `base_dir=` when supplied.
* Unknown keys raise `ConfigError` instead of being silently ignored.
* Every numeric field is range-checked at construction time.
* `config.to_dict()` returns a plain nested dictionary, suitable for logging.
* `config == AppConfig.from_dict(config.to_dict())` holds, so a config can be
  serialised and restored without drift.

## 2. Sections

| Section | Purpose | Key fields |
|---|---|---|
| `model` | Baseline model paths and vision threshold | `yolo_path`, `siren_classifier_path`, `confidence_threshold`, `imgsz` |
| `video` | Source selection, sampling and dashboard preview size | `source_type` (`file`/`webcam`/`rtsp`), `frame_skip`, `read_delay_s`, `reconnect_delay_s`, `read_timeout_s`, `display_width`, `display_height` |
| `audio` | Streaming windows and classifier threshold | `sample_rate`, `window_s`, `overlap_s`, `block_s`, `threshold`, `channels`, `device` |
| `signal` | Simulated state-machine timing and hysteresis | `activation_threshold`, `release_threshold`, `min_confirmation_s`, `release_grace_s`, `max_priority_s`, `min_green_s`, `yellow_s`, `all_red_s` |
| `fusion` | Evidence policy and event lifetime | `policy` (`VIDEO`/`AUDIO`/`BOTH`/`ADAPTIVE`), `stale_after_s`, `debounce_s`, `adaptive_window_s` |
| `logging` | Log destinations and level | `level`, `text_file`, `json_file` |
| `monitoring` | Metrics cadence | `metrics_interval_s` |
| `system` | Number and names of approaches | `num_approaches`, `approach_names` |

## 3. Safety-relevant validation

The configuration rejects values that would make the simulator unsafe or
ambiguous:

* `signal.release_threshold > signal.activation_threshold` is rejected, so
  hysteresis cannot be inverted.
* `signal.fail_safe_state` is limited to `ALL_RED`.
* `audio.overlap_s >= audio.window_s` is rejected.
* `video.rtsp_url` is mandatory when `video.source_type: rtsp`.
* `fusion.policy` is restricted to the four documented policies.
* `fusion.adaptive_window_s` must be positive and no larger than
  `fusion.stale_after_s`, so ADAPTIVE can never be looser than BOTH.
* Model image size must be a positive multiple of 32.

## 4. Event schema

`ambulance_system.events` defines the unit of evidence used by later stages:

```python
event = DetectionEvent.video(
    approach="Road 1",
    confidence=0.87,
    bbox=(10.0, 20.0, 110.0, 120.0),
    frame_index=42,
)
event.to_dict()   # JSON-serialisable, includes an ISO-8601 UTC timestamp
```

Fields: `event_id`, `timestamp`, `approach`, `source`, `detection_type`,
`confidence`, `status`, `evidence_refs`, plus modality-specific `bbox` /
`frame_index` or `window_start_s` / `window_end_s`.

`PriorityRequest` records the fusion outcome for one approach and carries the
evidence IDs that produced it. In Stage 8 the dashboard view-model converts
fusion decisions to `ApproachObservation` values for the Stage 3
`SignalController`; see [`dashboard.md`](dashboard.md).

## 5. Logging

```python
from ambulance_system.logging_utils import configure_logging, get_logger

configure_logging("INFO", json_file="logs/events.jsonl")
get_logger("vision").info("frame processed", extra={"approach": "Road 1"})
```

* One JSON object per line.
* Keys containing `password`, `secret`, `token`, `api_key`, `access_key`,
  `private_key` or `credential` are replaced with `***`, including when nested.
* `configure_logging` removes and closes previous handlers, so repeated calls
  in tests do not duplicate output.
* Importing the module has no side effects.

## 6. Running the tests

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Coverage (development-only dependencies in `requirements-dev.txt`):

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest -q --cov=src/ambulance_system --cov-report=term-missing
```

The complete suite is 351 tests with 92% statement coverage as measured on
2026-11-10; the per-module table and untested UI paths are recorded in
[`testing_report.md`](testing_report.md).

`pytest.ini` adds `src` to `pythonpath`; the package does not need to be
installed for tests or for `new.py` to keep working. `.coveragerc` limits the
coverage run to `src/ambulance_system` and shows missing lines.
