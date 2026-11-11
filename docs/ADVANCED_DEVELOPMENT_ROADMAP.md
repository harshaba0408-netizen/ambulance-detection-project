# Advanced Development Roadmap
## AI Ambulance Detection & Adaptive Traffic Signal Priority System

**Document status:** living implementation roadmap - Stage 11 implemented 2026-11-11 (content-group split); Stage 14 implemented 2026-11-12 (simulation/mock hardware boundary)
**Date:** 2026-11-09 (last updated 2026-11-12)
**Baseline commit:** `8b10f1bd79c5783a84aac8e313ba06a898cfa3d6` (`main` = `origin/main` at the start of Stage 14)
**Companion document:** [`BASELINE_AUDIT.md`](BASELINE_AUDIT.md) — all measured facts referenced here come from that audit.
**Source code changed so far:** Stages 2–14 are implemented. See each stage section for delivered files.

> This is a **software prototype and simulation**. Nothing in this roadmap authorises connection to public traffic infrastructure, physical relays or real signal controllers. Every stage keeps the system in simulation or, at most, an isolated low-voltage lab mock.

---

## 1. Purpose

At the baseline, the system worked: YOLOv8 detected ambulances, a Random Forest classified siren clips, a Tkinter dashboard drove four simulated approaches, and 16 tests passed.

It started as a prototype with five structural weaknesses:

1. **Not reproducible** — no training scripts, no retained splits, no `results.csv`. The published metrics cannot be regenerated.
2. **Not adaptive** — detection flags latch permanently and are cleared only by *Clear All*.
3. **Not streaming** — audio is classified once per file selection; video is file playback only.
4. **Not modular** — UI, inference, and signal policy live in one 441-line class, so business logic cannot be tested without Tkinter.
5. **Not measurable** — no coverage, no FPS/latency instrumentation, no benchmark harness.

This roadmap fixes those in dependency order, starting with the work that needs **no new data, no new hardware, and no expensive compute**.

---

## 2. Verified current state

### 2.1 Environment (hard constraint)

| Resource | Value | Consequence |
|---|---|---|
| CPU | Intel 4 logical cores | YOLOv8-L inference is CPU-only and slow; parallel workers are limited |
| RAM | 7.9 GB total, **~1.0 GB free** | At most one model process + one light worker at a time |
| Disk | **2.0 GB free** | **Blocks local retraining and large dataset downloads** |
| GPU | None (`torch 2.10.0+cpu`, CUDA unavailable) | Training on this machine is not viable for YOLOv8-L |
| Python | 3.12.5, venv present | Fine |

Missing optional packages: `sounddevice` / `pyaudio` (microphone capture), `pytest-cov` (coverage), `webrtcvad`, `resampy`.
Installed and usable: `librosa 0.11.0`, `soundfile 0.13.1`, `opencv 4.13.0`, `psutil 7.2.2`, `PyYAML 6.0.3`.

### 2.2 Repository state

| Item | Value |
|---|---|
| Branch | `main` @ `8bbb3da3c`, clean, tracking `origin/main` |
| Remote | `https://github.com/harshaba0408-netizen/ambulance-detection-project.git` |
| Legacy history | preserved locally at `backup/full-history` (`770bb13af`) |
| Published model | `yolov8l_ambulance.pt` via Git LFS (87,657,530 bytes) |
| Unpublished local-only artefacts | `Yovov8l_Amubalce_trained.pt`, 398 WAVs, 400 spectrogram PNGs, 12 videos, `venv/` |

### 2.3 Verified baseline metrics (measurement, not target)

| Modality | Metric | Value | Provenance |
|---|---|---:|---|
| Vision | Precision | 0.87388 | `train_metrics` at best epoch, checkpoint metadata |
| Vision | Recall | **0.69374** | same |
| Vision | mAP@50 | 0.74229 | same |
| Vision | mAP@50-95 | 0.40584 | same |
| Vision | Validation split | 10 images | `data/ambulance_images/data.yaml` |
| Vision | Test split | 5 images | too small for statistical claims |
| Audio | Resubstitution accuracy | 1.0000 | measured on all 400 clips, **not** validation |
| Audio | Retained split / training script | **none** | cannot be reproduced |
| Audio | Duplicate pairs | 4 inside `siren` | direct leakage risk |
| System | Automated tests | 16 passed / 0 failed | 40.53 s, this machine |
| System | Coverage | **unmeasured** | `pytest-cov` absent |

The vision metrics are the **best-epoch snapshot** stored in the checkpoint. `results.csv`, `best.pt`, and `last.pt` do not exist. The final recorded epoch was materially worse (precision 0.98872, recall 0.50000, mAP@50 0.59132), which is itself evidence that 10 validation images produce a noisy model-selection signal.

### 2.4 Baseline architecture (as implemented before Stage 2)

```
Tkinter main thread (new.py :: AmbulanceDetectionApp)
│
├── load_models()                 YOLO("yolov8l_ambulance.pt") + pickle.load(siren_classifier.pkl)
├── build_ui()                    4 approach cards, per-road file pickers, policy combo, canvases, log box
├── start_video()  ──► thread     cv2.VideoCapture → resize → ImageTk → root.after(0, update_video)
│                                 YOLO every 5th frame *only while video_detected[road] is False*
├── start_audio_detection() ──► thread
│                                 librosa.load(duration=3.0) → 40 MFCC → mean → RF.predict
├── result_queue ──► drain_queue() (Tk `after(100)`) sets latched booleans
└── recompute_signals()           policy gate (AUDIO|VIDEO|BOTH) → earliest first_detect_ts wins
                                  → set_light() red/green; no yellow, no timing, no release
```

Consequences that the refactor must remove:

* `video_detected` / `audio_detected` are **write-once**; `clear_all()` is the only reset path.
* After the first positive, the video worker **stops running YOLO for that road entirely**.
* `self.log()` is invoked from worker threads and writes to a Tk widget → thread-safety hazard.
* The 6 signal tests exercise a **copy** of the coordinator defined in the test file, not `new.py`.
* Confidence threshold exists twice: `new.py` 0.65, `tests/test_yolo_detection.py` 0.70.

---

## 3. Target architecture

```
                     ┌──────────────────────┐
                     │  config/default.yaml │  validated by dataclasses
                     └──────────┬───────────┘
                                │
   ┌────────────────┬───────────┴───────────┬────────────────┐
   │ VideoSource    │   AudioSource         │  Clock         │  (interfaces)
   │ file/webcam/   │   file / microphone   │  monotonic     │
   │ rtsp           │   (sounddevice)       │                │
   └───────┬────────┴───────────┬───────────┴────────────────┘
           │                    │
   ┌───────▼────────┐   ┌───────▼─────────┐
   │ Detector       │   │ SirenClassifier │
   │ (YOLO wrapper) │   │ (MFCC + RF)     │
   └───────┬────────┘   └───────┬─────────┘
           │ DetectionEvent             │ AudioEvent
           └────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │  FusionEngine    │  policy, hysteresis, staleness, dedupe
              └────────┬─────────┘
                       ▼  PriorityRequest
              ┌──────────────────┐
              │ SignalStateMachine│ NORMAL / PRIORITY / CLEARING / FAULT
              └────────┬─────────┘
                       ▼  SignalCommand + EventLog
              ┌──────────────────┐
              │ Dashboard (Tk)   │  reads snapshots via queue, never blocks
              └──────────────────┘
```

Rules the design must enforce:

* Business logic modules are **import-safe**: no module-level model loading, no `tk.Tk()`, no capture devices opened at import.
* The UI only renders snapshots and sends commands; it never owns inference state.
* Time is injected (a `Clock` callable) so state machines are testable without sleeping.
* Every decision emits a structured event with an ID, timestamp, approach, source, confidence and status.

---

## 4. Test status and quality gap

**Baseline at Stage 1:** 16 collected, 16 passed, 0 failed, 0 skipped, 1 Librosa deprecation warning, 40.53 s. **Current (2026-11-12):** 416 passed, 1 warning, 46.05 s plain run; 92% statement coverage (221 missed of 2778) measured with the 388-test suite.

| Area | Existing tests | Real coverage of production code | Gap |
|---|---:|---|---|
| Vision | 5 | Model load, synthetic frame, one real image, bad path | No video source, no filtering, no no-detection case |
| Audio | 5 | Load, MFCC shape, 2 file predictions, bad path | No streaming, silence, disconnection |
| Signal | 6 | **A copy of the class**, not `new.py` | No timing, release, conflict, fault, override |
| Fusion | 0 | — | Entirely untested |
| Dashboard | 0 | — | Entirely untested |

**Non-negotiable:** all 16 existing tests must remain and keep their assertions. New coverage is added alongside.

---

## 5. Development stages

Stages are ordered by dependency and by what is actually executable on this machine. Complexity is engineering effort for one experienced developer working alone on this hardware, **excluding** external data acquisition and training time.

Legend — effort: **S** ≤ 2 days, **M** 3–5 days, **L** 6–10 days, **XL** > 10 days or externally blocked.

---

### Stage 2 — Modular foundation: config, events, logging, clock
**Status:** implemented 2026-11-09 — 48 new tests, 64 total passing. **Complexity: S (1–2 days). Risk: low. Changes behaviour: none.**

Objective: create the skeleton every later stage plugs into, without touching runtime behaviour.

Delivered:

```
src/ambulance_system/__init__.py
src/ambulance_system/config.py          # dataclasses + YAML loader + validation
src/ambulance_system/events.py          # DetectionEvent, PriorityRequest
src/ambulance_system/clock.py           # Clock protocol + RealClock + FakeClock
src/ambulance_system/logging_utils.py   # structured JSON-lines logger, secret redaction
config/default.yaml                     # model paths, thresholds, windows, timings
docs/configuration.md
tests/test_config.py, test_events.py, test_clock.py, test_logging_utils.py
```

Modify: `pytest.ini` (`pythonpath = . src`, repo-local temp root), `.gitignore` (ignore `logs/`, `runs/`, `artifacts/`, `benchmarks/results/`, `.pytest_tmp/`). `requirements.txt` already contained `PyYAML`, so it was not changed. `SignalCommand` was deliberately deferred to Stage 3 so signal states are defined once.

Exit criteria:
* `config/default.yaml` round-trips into validated dataclasses; invalid values raise clear errors.
* Event schema has stable IDs, UTC timestamps, `source`, `approach`, `confidence`, `status`, `evidence_ref`.
* `FakeClock` allows deterministic time tests.
* All 16 existing tests still pass; no `new.py` change yet.

Explicitly **not** in this stage: rewriting `new.py`, changing thresholds, adding dependencies beyond already-installed `PyYAML`.

---

### Stage 3 — Signal state machine v2 (headless, fully tested)
**Status:** implemented 2026-11-09 — 45 new tests, 109 total passing. **Complexity: M (3–5 days). Risk: medium.**

Objective: replace latched booleans with an explicit, testable state machine — the highest-value functional change.

Create:

```
src/ambulance_system/traffic_signal/states.py      # NORMAL, PRIORITY, CLEARING, FAULT (+ RED/YELLOW/GREEN output)
src/ambulance_system/traffic_signal/machine.py     # per-approach machine, validated transitions
src/ambulance_system/traffic_signal/arbiter.py     # conflict resolution across 4 approaches
tests/test_signal_machine.py
tests/test_signal_arbiter.py
```

Behaviour to implement:
* Activation threshold **and** a lower release threshold (hysteresis).
* Minimum confirmation duration before priority is granted.
* Release grace period; priority does not drop on one missed frame.
* Maximum priority duration with forced `CLEARING`.
* Transition timing: GREEN → YELLOW → RED with configurable minimum green and clearance interval.
* Deterministic conflict policy when multiple approaches request priority; never two incompatible greens.
* `FAULT` state forces all-red and requires an explicit reset.
* Manual `Clear All` remains available and maps to an explicit transition.

Modify: nothing in `new.py` yet. `tests/test_traffic_signal_logic.py` stays untouched; the new tests are additive.

Delivered: `src/ambulance_system/traffic_signal/{states,machine,arbiter}.py`, `docs/signal_state_machine.md`, and two test modules. `SignalController` consumes `ApproachObservation` values so Stage 7 can supply fused evidence without changing the controller API.

Exit criteria: transition table covered by tests including illegal-transition rejection; zero real sleeps (FakeClock); 16 legacy tests unaffected.

---

### Stage 4 — Vision detection module + honest evaluation harness
**Status:** implemented 2026-11-09 — 23 new tests, 132 total passing. **Complexity: M (2–4 days) for code; XL for retraining (Stage 12).**

Objective: extract detection so it is testable and measurable, and remove threshold drift.

Create:

```
src/ambulance_system/detection/detector.py     # YOLO wrapper: load, predict, filter, metadata
src/ambulance_system/detection/types.py        # Detection dataclass (bbox, conf, class, ts)
scripts/evaluate_yolo.py                       # runs ultralytics val on a given split, writes JSON
tests/test_detector.py
```

Modify: `new.py::detect_ambulance` (delegate to the module, single `CONF_THRESHOLD` source), `requirements.txt` (no change needed), `docs/model_evaluation.md` (record that metrics are checkpoint-derived, not recomputed).

Delivered: `src/ambulance_system/detection/{detector,types}.py`, `scripts/evaluate_yolo.py`, `tests/test_detector.py`, `tests/test_evaluate_yolo.py`, and a `sys.path` bootstrap in `new.py`. `new.py` now loads thresholds and paths from `config/default.yaml` and delegates detection to `Detector`; `yolov8l_ambulance.pt` itself was not modified. Real test-split results at 640 and 320 are recorded in `docs/model_evaluation.md` and `benchmarks/results/` with explicit small-sample caveats. Plot generation had to be disabled (`--no-plots`) because it exceeded 15 minutes on this machine.

Evaluation output must include per-image detections and confidences, not just aggregate metrics, so misses are inspectable. Do **not** report the 5-image test split as a benchmark.

Exit criteria: detector unit-tested for empty frame, low-confidence box, class-name mismatch, missing weights; `new.py` and tests share one threshold constant.

---

### Stage 5 — Video source abstraction (file / webcam / RTSP) + pipeline metrics
**Status:** implemented 2026-11-09 — 46 new tests, 178 total passing. **Complexity: M (3–5 days).**

Objective: one interface for all video inputs, with sampling, reconnection and instrumentation.

Create:

```
src/ambulance_system/video/source.py       # VideoSource protocol; FileSource, WebcamSource, RTSPSource
src/ambulance_system/video/pipeline.py     # acquisition thread → bounded queue → inference worker
src/ambulance_system/monitoring/metrics.py # FPS, latency, dropped frames, CPU/mem via psutil
tests/test_video_source.py                 # synthetic frames, no device required
```

Modify: `new.py::start_video` / `worker` / `stop_video` (use the pipeline), `new.py` teardown path (release capture reliably).

Delivered: `src/ambulance_system/video/{source,pipeline}.py`, `src/ambulance_system/monitoring/metrics.py`, `tests/test_video_source.py`, `tests/test_video_pipeline.py`, `tests/test_metrics.py`, and `docs/video_pipeline.md`. `new.py` now uses `FileVideoSource` + `VideoPipeline`; `read_delay_s` was added to `VideoConfig` and `config/default.yaml`. File sources shut down deterministically, sampling is configured by `video.frame_skip`, and FPS/latency/drop counters are measured at runtime. Webcam and RTSP sources are implemented and unit-tested with injected captures but remain unvalidated on hardware or a live stream. `sounddevice` was not required. A measured run on `data/Video/1.mp4` (96 frames, CPU-only, queue size 2) delivered 21 frames, dropped 75, ran 4 sampled inferences in 11.36 s, with 525 ms mean and 2.72 s p95 callback latency; this is a reproducibility check, not a real-time claim. The harness is `scripts/benchmark_video_pipeline.py`.

RTSP must remain **configurable but unverified** until an authorised stream is supplied. Tests use a synthetic source, never the network.

Exit criteria: clean shutdown releases the device; frame sampling configurable (process every N frames); measured FPS/latency recorded, not estimated.

---

### Stage 6 — Continuous audio streaming
**Status:** offline file path implemented 2026-11-09 — 31 new tests. Live microphone path still blocked on `sounddevice`. **Complexity: L (5–8 days).**

Objective: run the existing classifier on a rolling window instead of one file.

Create:

```
src/ambulance_system/audio/features.py     # MFCC extraction, single implementation
src/ambulance_system/audio/classifier.py   # wraps siren_classifier.pkl, exposes predict_proba
src/ambulance_system/audio/streaming.py    # ring buffer, overlap, worker thread, bounded queue
src/ambulance_system/audio/devices.py      # MicrophoneSource (sounddevice) + FileStreamSource
tests/test_audio_streaming.py              # file-backed stream, silence, clipped audio
```

Modify: `new.py::start_audio_detection` (replace one-shot with streaming worker), `requirements.txt` (**approval required** for `sounddevice`).

Delivered: `src/ambulance_system/audio/{features,classifier,devices,streaming}.py`, `tests/test_audio_features.py`, `tests/test_audio_classifier.py`, `tests/test_audio_streaming.py`, and `docs/audio_pipeline.md`. `new.py` now scans every sliding window of the selected file instead of only the first three seconds, uses a single MFCC implementation that is tested against the legacy calculation, and consumes positive-class probabilities when available. The file source yields mono blocks; live microphone capture is not implemented and no audio dependency was installed. Real checks on two 3-second clips produced two windows each and the expected siren/non-siren flags; first-call librosa warm-up takes about 8 s on this machine, later files a few hundred milliseconds.

Keep the existing classifier and its 40-MFCC pipeline. `predict_proba` gives a real confidence signal for fusion; the current code only exposes a hard label.

Exit criteria: 1–3 s window with configurable overlap; no blocking of the video path; timestamps on every prediction; graceful handling of device loss; existing audio tests untouched.

**Approval gate:** installing `sounddevice` (small wheel, but it is a new dependency and opens a microphone device).

---

### Stage 7 — Multimodal fusion engine
**Status:** implemented 2026-11-09 — 35 new tests. Wired into the dashboard in Stage 8. **Complexity: M (3–5 days).**

Objective: one component decides whether an approach has an emergency, instead of scattered booleans.

Create:

```
src/ambulance_system/fusion/engine.py      # policies VIDEO / AUDIO / BOTH / ADAPTIVE
src/ambulance_system/fusion/rules.py       # temporal confirmation, hysteresis, staleness, dedupe
tests/test_fusion.py
```

Rules:
* Never average audio and video confidence — they are not commensurable.
* `ADAPTIVE` is configurable and must not silently reduce confirmation requirements; document every rule.
* Events expire after a configurable staleness window.
* Duplicate suppression by `(approach, source, type)` within a debounce interval.
* Every state change produces an event with the evidence IDs that caused it.

Exit criteria: table-driven tests for ALL policy × evidence combinations, stale expiry, duplicate suppression, and conflicting simultaneous requests.

Delivered: `src/ambulance_system/fusion/{engine,rules}.py`, `tests/test_fusion.py`, and `docs/fusion.md`. `FusionConfig.adaptive_window_s` was added and validated to be no larger than `stale_after_s`; ADAPTIVE requires both modalities inside that tighter window and uses the same conservative minimum confidence as BOTH. Video and audio confidences are never averaged. Duplicate suppression uses `(approach, source, type)` and event timestamps; expired events are pruned and counted. `FusionDecision.to_observation()` feeds the Stage 3 controller, and a test confirms that simultaneous fusion requests still produce exactly one GREEN. Dashboard wiring was deliberately deferred to Stage 8, where it was completed.

---

### Stage 8 — Dashboard upgrade and thread safety
**Status:** implemented 2026-11-10 - 29 new tests, 273 total at that stage (388 now). **Complexity: L (5-8 days).**

Objective: keep the existing Tkinter UI but make it a thin, non-blocking renderer.

Create:

```
src/ambulance_system/dashboard/app.py      # Tk app, snapshot rendering only
src/ambulance_system/dashboard/viewmodel.py# immutable snapshot built off-thread
src/ambulance_system/dashboard/widgets.py  # signal canvas, event log, health panel
tests/test_viewmodel.py
```

Modify: `new.py` (becomes a thin entry point or is superseded by `src/.../dashboard/app.py` — decide at implementation time, keep a working launch command), `README.md` run instructions.

Add: 4-approach visualisation, live preview, boxes, audio/video confidence, current state, event history with timestamps, FPS/latency, system health, error notifications, manual reset, start/stop.

All Tk calls stay on the main thread via `after()`; workers only push snapshots. Remove `self.log()` calls from worker threads.

Exit criteria: dashboard remains responsive during inference; UI logic tested through the view-model without a display where possible; legacy Clear All still works.

Delivered: `src/ambulance_system/dashboard/{__init__,viewmodel,widgets,app}.py`, `tests/test_viewmodel.py`, `docs/dashboard.md`, and a thin `new.py` entry point. The package root exports only headless view-model types, so importing it does not import Tkinter. `app.py` owns the video pipelines and audio workers, pushes evidence and errors into the view-model, and renders immutable snapshots on the Tk main thread every 100 ms via `after()`. Worker threads no longer call Tk logging methods. The four approach cards show preview with bounding boxes, audio/video confidence, fused state and reason, signal head, phase, FPS, latency and dropped frames; the window also exposes measured CPU/RAM health, a timestamped bounded event history, error notifications, Simulate/Clear Fault, per-approach Start/Stop, per-approach AUDIO/VIDEO/BOTH/ADAPTIVE policy selection, and the legacy Clear All. The fusion engine gained additive per-approach policies (`set_policy`, `policy_for`, `policies`, `reset()`). `tests/test_viewmodel.py` adds 27 headless tests (18 functions plus 10 parameterised invalid-input cases); `tests/test_fusion.py` adds 2. Full suite at Stage 8 completion: **273 passed, 1 warning in 37.63 s**. A manual UI smoke test with a withdrawn root, fake detector/classifier and `app.on_close` also passed; it is not part of the repository test suite. Webcam/RTSP dashboard use and live microphone capture remain unvalidated.

---

### Stage 9 — Test expansion, coverage, CI-style checks
**Status:** implemented 2026-11-10 — 78 new tests, 351 total passing, 92% statement coverage. **Complexity: M (2–3 days).**

Create/modify: `tests/test_detection.py`, `test_audio.py`, `test_video.py`, `test_fusion.py`, `test_signal.py`, `test_dashboard.py`, `pytest.ini`, `requirements-dev.txt`, `.github/workflows/tests.yml` (optional, only if you want CI).

Run and report exactly:

```
python -m pytest -v
python -m pytest --cov=src/ambulance_system --cov-report=term-missing
```

Exit criteria: 16 legacy tests still present and passing; new tests cover every module listed in section 4; coverage reported honestly, including untested modules; no test weakened or skipped to force green.

Delivered: installed `pytest-cov==7.1.0` / `coverage==7.17.2` with approval; added `requirements-dev.txt` and `.coveragerc`. Added 78 tests across `tests/test_events_validation.py`, `tests/test_dashboard_app.py`, `tests/test_dashboard_widgets.py`, expanded `tests/test_config.py` and `tests/test_signal_machine.py`, and added `tests/conftest.py` for one hidden session Tk root. The existing module test files were kept rather than renamed to the planned `test_detection.py` / `test_audio.py` / `test_video.py` names, because they already cover those modules and renaming would churn history. No GitHub Actions workflow was added. Measured result: **351 passed, 1 warning; 92% statement coverage (221 missed of 2778)**. `dashboard/widgets.py` is 100%, `events.py` 100%, `dashboard/viewmodel.py` 97%, `config.py` 96%; `dashboard/app.py` is 68%, with the gaps being model loading, file dialogs, real worker threads and `main()`. The full per-module table is in [`testing_report.md`](testing_report.md) section 6. Tk rendering tests skip with an explicit reason only when no display exists; on this machine they all run. No legacy test was modified, weakened or skipped.

---

### Stage 10 — Benchmark harness and baseline comparison
**Status:** implemented 2026-11-10 - 16 new tests, 367 total at that stage (388 now). **Complexity: M (2-4 days).**

Create:

```
scripts/benchmark_vision.py
scripts/benchmark_audio.py
scripts/benchmark_system.py
docs/BASELINE_VS_IMPROVED.md
benchmarks/results/<timestamp>.json
```

Measure on this machine and record the hardware string with every result: vision precision/recall/mAP/latency/FPS; audio precision/recall/F1/FPR/latency; system end-to-end latency, event latency, transition latency, CPU, RAM.

Exit criteria: every number traceable to a stored JSON result and a command; no estimated values presented as measured; identical hardware/config noted for any comparison.

Delivered: `scripts/benchmark_common.py` (shared hashing, ratio, percentile, latency, environment and JSON helpers), `scripts/benchmark_vision.py`, `scripts/benchmark_audio.py`, `scripts/benchmark_system.py`, `tests/test_benchmark_harness.py` and [`BASELINE_VS_IMPROVED.md`](BASELINE_VS_IMPROVED.md). Stored results: `benchmarks/results/vision_test_640_20261110T084458Z.json`, `benchmarks/results/audio_resubstitution_20261110T083904Z.json` and `benchmarks/results/system_simulation_20261110T085500Z.json`. Vision test-split metrics (5 images, `conf=0.001`): precision 0.9207, recall 0.4000, mAP@50 0.5217, mAP@50-95 0.4088; validation-loop pipeline 2632.54 ms/image (0.3799 images/s) on this CPU. Audio: 200 siren + 200 non-siren clips, resubstitution accuracy/precision/recall/F1 1.0000, FPR 0.0000, median end-to-end latency 173.21 ms; four exact-duplicate pairs are reported by hash. System: synthetic evidence through `DashboardViewModel` + `FakeClock`, median `tick()` wall latency 0.0972 ms, transition ticks 0.3185 ms mean, simulated activation delay 2.0 s and yellow+all-red clearance 5.0 s; process RAM delta under 0.04 MB. No checkpoint was retrained or replaced, so the comparison document explicitly labels vision differences as protocol/split differences, audio as resubstitution, and system timings as software decision-path measurements with no model inference. None of the stored results claims real-world traffic-control performance.

---

### Stage 11 - Audio data remediation and provenance
**Status:** implemented 2026-11-11; the follow-up content-group split holds out 234 acoustic groups, but source provenance is still unverified and needs your input. **Complexity: M (2-3 days) + external data.**

Delivered: `scripts/audio_integrity.py`, `scripts/audio_content_groups.py`, `scripts/make_audio_split.py`, `scripts/train_audio.py`, `tests/test_audio_provenance.py` (31 tests), `data/audio/content_groups.csv`, `data/audio/splits/audio_split.json` and [`AUDIO_DATASET_PROVENANCE.md`](AUDIO_DATASET_PROVENANCE.md). Stored results: `benchmarks/results/audio_integrity_20261111T071208Z.json`, `benchmarks/results/audio_content_groups_20261111.json` and `benchmarks/results/audio_training_20261111T181213Z.json`.

Findings: 400 files with no numbering gaps; four exact siren duplicate pairs (`73 = 76`, `74 = 77`, `75 = 78`, `129 = 130`); mixed siren formats and four `ISFT` encoder groups; no source, session, recording or licence metadata. The audit is read-only and no quarantine move was performed.

Provisional split: seed `stage11-20261110`, 396 retained files after excluding the four duplicates, split 277 / 59 / 60 (train / val / test). That file-level result is duplicate-contaminated.

Content-group follow-up (the Stage 12 audio work): seed `stage12-20261111`, 234 acoustic groups (60 non-siren, 174 siren), 0 groups span splits, split 280 / 62 / 54. The new Random Forest is labelled `held_out_content_group` and scored 1.0000 on validation/test, but that is **not** source-independent. The shipped `siren_classifier.pkl` was not replaced; experiments live under `artifacts/experiments/`.

Exit criteria met for reproducibility: the split file is generated from paths or a content-group manifest (not audio) and ready to commit, the same command reproduces the split and model given the same data, and content-group-held-out metrics are reported. Source-independent mode is implemented through `--manifest` and stays blocked until verified session/source metadata exists.

---

### Stage 12 — Vision dataset expansion and YOLO retraining
**Status: blocked on this machine.** **Complexity: XL. Risk: high.**

Blockers, all verified: 2.0 GB free disk, no GPU, 4 CPU cores, ~1 GB free RAM, no licensed extra data, no training script.

Create once resources exist: `docs/DATASET_PROVENANCE.md`, `scripts/prepare_dataset.py` (dedupe, near-duplicate check, leakage check, split generation), `scripts/train_yolo.py`, `runs/experiments/<id>/`.

Plan:
* Verify licence and redistribution terms **before** download.
* Add diversity: night, rain, occlusion, distance, front/rear, multiple ambulance liveries.
* Keep the 5-image test split out of tuning; build a new, larger held-out test set.
* Train at a smaller image size or with a smaller model (e.g. YOLOv8s/m) if CPU-only comparison is the only option; do not assume larger is better.
* Preserve `yolov8l_ambulance.pt` untouched; all new weights go under `runs/experiments/`.
* Target: higher recall at acceptable false-positive cost. Do not promise a number.

**Approval gate:** dataset download, training time, and disk space must be approved explicitly.

---

### Stage 13 — Optional GCC-PHAT direction research
**Status:** optional and independent. **Complexity: L (5–8 days). Risk: research.**

Create only if multi-channel audio or a mic array becomes available:

```
src/ambulance_system/localisation/gcc_phat.py
src/ambulance_system/localisation/doa.py
tests/test_gcc_phat.py
docs/localisation.md
```

Deliverable is a self-contained offline experiment: mathematical derivation, synthetic two-source simulation, unit tests with known delays, and an explicit limitation section. It must not be wired into the priority decision and must never claim geographic localisation from single-microphone classification.

---

### Stage 14 — Hardware abstraction (simulation + mock only)
**Status:** implemented 2026-11-12. **Complexity: M (2–3 days).**

Delivered: `src/ambulance_system/hardware/controller.py` (interface, safe defaults, `HardwareCommand`), `simulation.py` (default in-memory controller with timestamped history), `mock.py` (programmable test double), `tests/test_hardware.py` (18 tests) and [`hardware_integration.md`](hardware_integration.md).

Every controller starts disconnected and all-red, forces red on connect/disconnect, rejects commands while disconnected, and leaves the recorded state unchanged if a transport write fails. A source-scan test rejects device-transport imports.

No GPIO, relay, serial or network device is opened by default. The abstraction is not wired into the dashboard. Any future physical prototype must be isolated low-voltage lab equipment with manual supervision and an independent emergency shutdown. **No physical hardware has been tested.**

---

### Stage 15 — Documentation, README, release
**Status:** final. **Complexity: M (2–4 days).**

Update `README.md`, `docs/architecture.md`, `docs/testing_report.md`, `docs/dataset_report.md`; add `docs/limitations.md` and the final research report. Label explicitly: **Implemented**, **Tested**, **Experimental**, **Planned**. Generate CV bullets only from completed, verified work.

Git: work on `feature/advanced-prototype` (or similar), meaningful commits, no force push, no secrets, models via LFS only after checking quota, and **ask before every commit and push**.

---

## 6. Exact files that need modification

| File | Stage(s) | Change | Risk |
|---|---|---|---|
| `new.py` | 4, 5, 6, 8 | Delegation to modules; thread-safe logging; pipeline wiring; eventually thin entry point | **High** — primary working app |
| `ambulance_and_siren_integration.py` | 4, 6 | Remove module-level side effects; guard under `__main__` | Medium |
| `ambulance_test.py` | 4 | Use shared detector/threshold | Low |
| `test_siren.py` | 6 | Use shared feature extractor | Low |
| `tests/test_yolo_detection.py` | 4 | Keep all assertions; import shared threshold constant | Low |
| `tests/test_traffic_signal_logic.py` | 3 | **Preserve unchanged**; add parallel machine tests | Low |
| `tests/test_audio_classification.py` | 6 | **Preserve unchanged**; add streaming tests | Low |
| `pytest.ini` | 2, 9 | `pythonpath`, coverage options | Low |
| `requirements.txt` | 2, 6, 9 | Add `PyYAML`, `sounddevice`, `pytest-cov` (each approved) | Low |
| `.gitignore` | 2, 10, 12 | `logs/`, `benchmarks/results/`, `runs/`, `artifacts/` | Low |
| `README.md` | 8, 15 | Run commands, features, status labels | Low |
| `docs/architecture.md` | 8, 15 | Replace as-is diagram with target | Low |
| `docs/model_evaluation.md` | 4, 12 | Clarify checkpoint-derived metrics; add new results | Low |
| `docs/testing_report.md` | 9, 15 | Real counts and coverage | Low |
| `docs/dataset_report.md` | 11, 12 | Split and provenance accuracy | Low |

New directories: `src/ambulance_system/`, `config/`, `scripts/`, `benchmarks/`, `runs/` (ignored), `artifacts/` (ignored).

There is a duplicate local checkpoint `Yovov8l_Amubalce_trained.pt` (87.6 MB, different hash). It is **not** referenced by code. Do not delete it without your approval; at most add it to `.gitignore` so it can never be committed.

---

## 7. Required external resources and approval gates

| Need | Blocks | Cost / size | Approval |
|---|---|---|---|
| `sounddevice` package | Live microphone capture (Stage 6) | Small wheel | **Required** |
| `pytest-cov` package | Coverage reporting (Stage 9) | Small wheel | **Approved and installed 2026-11-10** |
| ≥ 10–20 GB free disk | Dataset expansion + training artefacts (Stage 12) | — | **Required** |
| GPU or paid compute | Practical YOLOv8-L retraining | Cloud cost | **Required** |
| Licensed ambulance dataset | Vision improvement (Stage 12) | Download size varies | **Required** |
| Verified audio provenance / replacement set | Audio reproducibility (Stage 11) | — | **Required** |
| Authorised RTSP test stream | Verified live streaming (Stage 5) | — | **Required** |
| Optional 2-mic array | GCC-PHAT research (Stage 13) | Hardware purchase | **Required** |

Stage 10 is complete. Stage 11 tooling and the Stage 12 audio follow-up are implemented; source provenance remains unverified. Stage 14 (hardware simulation/mock boundary) is implemented. Stage 15 docs/release and the optional Stage 13 research remain; the vision Stage 12 retraining remains resource-blocked. `pytest-cov` is installed and no longer a blocker.

---

## 8. Traceability to the original phase list

| Original phase | Roadmap stage |
|---|---|
| 1 — Audit and baseline | Stage 1 (complete) |
| 2 — YOLOv8 improvement | Stage 12 (blocked) + Stage 4 (code/eval) |
| 3 — Continuous audio | Stage 6 (+ Stage 11 for data) |
| 4 — Live video | Stage 5 |
| 5 — Siren localisation | Stage 13 (optional) |
| 6 — Adaptive signal engine | Stage 3 |
| 7 — Multimodal fusion | Stage 7 |
| 8 — Dashboard | Stage 8 |
| 9 — Testing / QA | Stage 9 |
| 10 — Benchmarking | Stage 10 |
| 11 — Architecture | Stages 2–8 |
| 12 — Hardware-in-the-loop prep | Stage 14 |
| 13 — Config / reproducibility | Stage 2, 11, 12 |
| 14 — Final documentation | Stage 15 |
| 15 — Git / release | Stage 15 (all stages: branch + review) |
| 16 — Research / CV report | Stage 15 |

---

## 9. Working agreement for every stage

1. Inspect the affected files before editing.
2. State the design and the file list before implementation.
3. Implement the smallest change that satisfies the stage.
4. Run `python -m pytest -q`; all 16 legacy tests must still pass.
5. Compare against the recorded baseline where a metric is affected.
6. Update the relevant document in the same change.
7. Report: what changed, what was measured, what is still blocked.
8. **Ask before** deleting files, replacing weights, downloading datasets, training, installing major dependencies, committing, or pushing.

Quality rules:

* No fabricated metrics. Every number in a document must come from a command that can be re-run.
* No coverage claim without `pytest-cov` output.
* No "real-time" or "production-ready" claim unless measured on stated hardware.
* The simulator never claims traffic-safety certification.
* `docs/BASELINE_AUDIT.md` remains the frozen reference for the starting state.

---

## 10. Recommended order and first step

```
Stage 2  config + events + logging + clock      ← complete
Stage 3  signal state machine + tests           ← complete
Stage 4  detector module + eval harness         ← complete
Stage 5  video source abstraction               ← complete
Stage 6  audio streaming                        ← offline path complete; live mic needs sounddevice
Stage 7  fusion engine                            ← complete
Stage 8  dashboard upgrade                       ← complete
Stage 9  coverage + expanded tests              ← complete
Stage 10 benchmarks + comparison report          ← complete
Stage 11 audio data remediation                 ← implemented (content-group split; provenance unverified)
Stage 12 vision retraining                      ← blocked: disk / GPU / licensed data
Stage 13 GCC-PHAT (optional)
Stage 14 hardware abstraction (simulation only)  ← complete
Stage 15 docs, release, research report
```

**Current status: Stage 11 implemented (388 passing; file-level audio split and experiment metrics stored).** Source-independent audio metrics still require a provenance manifest. The next executable work is Stage 14 (simulation-only hardware abstraction) and Stage 15 documentation; Stage 12 is blocked by disk/GPU/licensed data, and Stage 13 needs optional microphone hardware. Release actions (commit/push) still require explicit approval.
