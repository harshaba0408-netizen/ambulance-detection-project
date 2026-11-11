# Automated Testing & Quality Assurance Report
## AI Ambulance Detection & Traffic Signal Priority System

> **Baseline document (Stage 1).** The tables below describe the original
> 16-test suite and remain as the frozen starting point. The current suite has
> **398 tests** (398 passed, 1 warning on 2026-11-11; the last coverage run with
> the 388-test suite reported **92% statement coverage**, 221 missed of 2778
> statements). See section 6 for the Stage 9 coverage report, section 7 for the
> Stage 10 benchmark-harness tests, section 8 for the Stage 11–12 provenance
> tests, and
> [`ADVANCED_DEVELOPMENT_ROADMAP.md`](ADVANCED_DEVELOPMENT_ROADMAP.md) for the
> stage status.

---

## 1. Test Suite Summary

An automated test suite using **pytest** was developed to independently validate model integrity, inference correctness, audio feature extraction, and traffic signal prioritization logic.

| Suite Category | File | Tests Count | Status |
| :--- | :--- | :--- | :--- |
| **YOLOv8 Detection** | `tests/test_yolo_detection.py` | 5 | **5 / 5 PASSED** |
| **Audio Classification** | `tests/test_audio_classification.py` | 5 | **5 / 5 PASSED** |
| **Signal Coordination Logic** | `tests/test_traffic_signal_logic.py` | 6 | **6 / 6 PASSED** |
| **Total** | — | **16** | **16 / 16 PASSED (100%)** |

**Execution Time:** 26.89 s (warm cache) / 68.73 s (cold cache) across two runs on 2026-11-09, CPU-only Windows with Python 3.12.5. Runtime varies by hardware.

**Scope:** These are automated unit/integration tests. No hardware-in-the-loop (GPIO/relay/Arduino) testing and no live RTSP traffic-camera testing was performed. The signal-coordination tests exercise a headless mirror of the decision rules (`TrafficSignalCoordinator` in the test file), not the Tkinter UI layer itself.

---

## 2. Test Cases Breakdown

### A. YOLOv8 Detection (`tests/test_yolo_detection.py`)
1. `test_yolo_model_file_exists`: Validates the physical presence of `yolov8l_ambulance.pt`.
2. `test_yolo_model_loading`: Confirms Ultralytics can instantiate the model and verify class mapping `0: ambulance`.
3. `test_yolo_inference_on_synthetic_frame`: Confirms inference runs successfully on blank $640\times 640$ tensors without memory corruption.
4. `test_yolo_inference_on_real_ambulance_image`: Runs inference on a test split ambulance image, confirming bounding box creation and confidence score $\ge 0.70$.
5. `test_yolo_invalid_model_path`: Confirms graceful error raising on invalid model paths.

### B. Audio Classification (`tests/test_audio_classification.py`)
1. `test_siren_classifier_exists`: Validates the presence of `siren_classifier.pkl`.
2. `test_siren_classifier_loading`: Confirms scikit-learn loads the Random Forest model and exposes the binary decision boundary.
3. `test_mfcc_feature_extraction_dimension`: Confirms Librosa extracts exactly $40$ MFCC coefficients with shape `(40,)` without NaN or Inf values.
4. `test_siren_classification_prediction`: Tests end-to-end prediction, verifying `1` (Siren) on emergency samples and `0` (Non-Siren) on environmental samples.
5. `test_invalid_audio_path_handling`: Verifies exception handling when an audio file cannot be loaded.

### C. Traffic Signal Decision Engine (`tests/test_traffic_signal_logic.py`)
1. `test_default_all_red`: Ensures all 4 intersection signals initialize to RED.
2. `test_audio_policy_triggers_green`: Validates that siren detection triggers a GREEN signal under the `AUDIO` policy while other approaches remain RED.
3. `test_video_policy_triggers_green`: Validates that ambulance visual detection triggers a GREEN signal under the `VIDEO` policy.
4. `test_both_policy_requires_audio_and_video`: Verifies dual-policy fail-safes (signal remains RED if only one sensor triggers, switches to GREEN only when both trigger simultaneously).
5. `test_revert_to_red_on_exit`: Confirms signal safely reverts to RED when the emergency vehicle clears.
6. `test_clear_all_resets_all_signals`: Confirms the global emergency reset function properly restores all 4 approaches to default status.

---

## 6. Stage 9 — Coverage and Expanded Tests

> This section is the Stage 9 snapshot (351 tests). The Stage 10 rerun is in
> section 7; totals and coverage are unchanged, and `video/pipeline.py` is
> currently 14 missed / 93% (see the rerun note there).

`pytest-cov` 7.1.0 and `coverage` 7.17.2 were installed with explicit approval;
they are development-only dependencies in `requirements-dev.txt`.

Command:

```powershell
.\venv\Scripts\python.exe -m pytest -q --cov=src/ambulance_system --cov-report=term-missing
```

Measured on 2026-11-10, Python 3.12.5, CPU-only Windows:

```
351 passed, 1 warning in 53.92s
TOTAL   2778 statements   221 missed   92%
```

### 6.1 Module coverage

All package `__init__.py` files are at 100% and are omitted from the table.

| Module | Statements | Missed | Coverage |
|---|---:|---:|---:|
| `__init__.py` | 2 | 0 | 100% |
| `audio/classifier.py` | 101 | 8 | 92% |
| `audio/devices.py` | 84 | 13 | 85% |
| `audio/features.py` | 32 | 0 | 100% |
| `audio/streaming.py` | 149 | 10 | 93% |
| `clock.py` | 31 | 2 | 94% |
| `config.py` | 238 | 10 | 96% |
| `dashboard/app.py` | 294 | 95 | 68% |
| `dashboard/viewmodel.py` | 304 | 10 | 97% |
| `dashboard/widgets.py` | 140 | 0 | 100% |
| `detection/detector.py` | 103 | 8 | 92% |
| `detection/types.py` | 36 | 5 | 86% |
| `events.py` | 118 | 0 | 100% |
| `fusion/engine.py` | 160 | 5 | 97% |
| `fusion/rules.py` | 84 | 3 | 96% |
| `logging_utils.py` | 59 | 5 | 92% |
| `monitoring/metrics.py` | 81 | 3 | 96% |
| `traffic_signal/arbiter.py` | 144 | 9 | 94% |
| `traffic_signal/machine.py` | 129 | 6 | 95% |
| `traffic_signal/states.py` | 97 | 3 | 97% |
| `video/pipeline.py` | 210 | 14 | 93% |
| `video/source.py` | 160 | 12 | 92% |

### 6.2 New Stage 9 tests

* `tests/test_events_validation.py` — malformed source/type/status, confidence,
  timestamp, IDs, evidence references, frame index, bbox and audio-window
  validation; `from_dict` and `PriorityRequest` rejection paths.
* `tests/test_config.py` — type-conversion errors (mapping, string, int, bool,
  float, sequence), integer/float coercion and empty-mapping defaults.
* `tests/test_signal_machine.py` — additional invalid transition cases,
  same-phase no-ops, and validation/serialisation for `ApproachObservation`,
  `SignalCommand` and `ApproachSnapshot`.
* `tests/test_dashboard_app.py` — `FrameStore` resize/bbox/BGR/grey/RGBA/clear
  behaviour, dashboard construction with injected fakes, policy/source sync,
  start/stop warnings, fault actions, missing video handling and close.
* `tests/test_dashboard_widgets.py` — signal canvas, approach card, event-log
  append/dedupe/trim, health panel and formatting helpers.
* `tests/conftest.py` — one hidden session Tk root and a fresh `Toplevel` per
  UI test; Tk tests skip with an explicit reason if no display is available.

### 6.3 Honest gaps

`dashboard/app.py` remains at 68%. The uncovered lines are real-model loading,
file dialogs, live video/audio worker bodies and `main()`. Those paths need a
display, model files and real media, and would be slow or flaky as unit tests;
they are covered by the manual smoke test only. Coverage is statement coverage,
not branch coverage, and says nothing about traffic-control correctness.

---

## 7. Stage 10 — Benchmark Harness Tests

Stage 10 added `tests/test_benchmark_harness.py`, 16 tests that run without
loading YOLO, the pickled classifier, a display, or the full audio/video
datasets.

Covered:

* `scripts/benchmark_common.py` — ratio with zero denominator, percentile index
  rule and bounds, millisecond latency summary, empty input, SHA-256, JSON
  writing with parent-directory creation.
* `scripts/benchmark_vision.py` — pipeline-stage sum excluding `loss`, missing
  stage handling, unmeasured FPS stays `None`, small-sample caveat.
* `scripts/benchmark_audio.py` — perfect and all-zero binary metric tables,
  duplicate-content grouping by hash.
* `scripts/benchmark_system.py` — transition detection and a deterministic
  4-second synthetic scenario that must grant green at simulated 2.0 s.

The system test uses the real `DashboardViewModel`, `FusionEngine` and
`SignalController` with `FakeClock`; it does not sleep or depend on wall time.
It asserts the simulated activation delay, the single transition and the absence
of resource sampling when `resources=None`.

Measured on 2026-11-10 after adding these tests:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

```
367 passed, 1 warning in 66.97s
```

Coverage rerun:

```powershell
.\venv\Scripts\python.exe -m pytest -q --cov=src/ambulance_system --cov-report=term-missing
```

```
367 passed, 1 warning in 94.80s
TOTAL   2778 statements   221 missed   92%
```

The coverage denominator is unchanged because the new tests cover the harness
under `scripts/`, which is outside `--cov=src/ambulance_system`. The rerun
reported `video/pipeline.py` as 14 missed / 93%; the Stage 9 table row above has
been corrected to match. No legacy test was modified, skipped or weakened.

---

## 8. Stage 11–12 — Audio Provenance Tests

Stage 11 added `tests/test_audio_provenance.py`; Stage 12 extended it to 31
tests. They use synthetic records, tiny temporary manifests and one hand-built
RIFF header for the unit tests. The only dataset-facing test reads the generated
`data/audio/content_groups.csv`; no test scans, loads or moves the real audio.

Covered:

* `scripts/audio_integrity.py` — filename-number parsing, duplicate grouping and
  canonical-copy selection, quarantine plan construction, format/encoder cohort
  boundaries, missing-number detection, unverified-provenance reporting, and
  `ISFT` extraction from a minimal RIFF file.
* `scripts/make_audio_split.py` — allocation edge cases, exact-duplicate
  exclusion, deterministic output for a fixed seed, different output for a
  different seed, per-class counts, CSV/JSON manifest loading, session groups
  staying inside one split, unmapped-file rejection, missing session/source
  rejection, class-spanning group rejection, and path normalisation. Stage 12
  adds `group_id` precedence, content-group mode detection, the
  `held_out_content_group` label, and the rule that a content-group split is
  never source-independent.
* `scripts/audio_content_groups.py` — union-find determinism, shifted-copy
  alignment, exact-correlation sanity, synthetic non-siren grouping, the
  Freesound 87130 reference union, and completeness of the generated CSV
  (396 rows, 60 non-siren groups, 174 siren groups, no class-spanning group).

Manifest enforcement is tested without any claim that a manifest is truthful:
the tests prove the generator enforces the supplied grouping, not that the
grouping reflects real recording sessions or sources.

Measured on 2026-11-11:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

```
398 passed, 1 warning in 42.38s
```

Coverage rerun (388-test suite; the Stage 12 additions are script/CSV tests and
do not change the `src/ambulance_system` coverage figures):

```powershell
.\venv\Scripts\python.exe -m pytest -q --cov=src/ambulance_system --cov-report=term-missing
```

```
388 passed, 1 warning in 50.23s
TOTAL   2778 statements   221 missed   92%
```

The new tests cover `scripts/`, so the `src/` statement denominator stays at
2778. No legacy test was modified, skipped or weakened.

