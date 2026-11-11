# Baseline vs Current Measurements

**Scope:** Stage 10 benchmark harness for the AI Ambulance Detection & Adaptive
Traffic Signal Priority prototype.

This document compares the **published baseline commit** with the **current
uncommitted working tree**. It reports only values that have a stored JSON result
and a command that produced it. Where the baseline never measured something, the
comparison says so instead of inventing a number.

> This is a software prototype and simulation. Nothing here is a claim about
> real traffic control, emergency-response times, or deployment readiness.

---

## 1. Ground rules

* The model checkpoints were **not** retrained, fine-tuned or replaced in
  Stages 2–10. Vision weights remain `yolov8l_ambulance.pt` (SHA-256
  `6784b16f90013517a55aec759710d7de1f87a468a8e0be9b9c3833588700229d`).
  The siren classifier remains `siren_classifier.pkl` (SHA-256
  `2aa5a0dfd561dc0ba9d5a80873b65b39f979675079286c698a85c5e41f3a8e5c`).
* Therefore **no metric below is evidence of model improvement**. Differences
  in vision metrics come from evaluation protocol and split, not from a better
  network.
* Audio numbers are **resubstitution**: the shipped classifier was trained on
  the same 400 clips. They are not held-out validation.
* System timings measure the Python decision path with synthetic evidence and a
  `FakeClock`. No camera, microphone, YOLO inference or audio classification is
  part of that measurement.
* Vision and system speed values are single-machine wall-clock observations.
  They vary with CPU load; they are not guarantees.

---

## 2. Hardware and software

All Stage 10 measurements were taken on the same machine:

| Item | Value |
|---|---|
| CPU | Intel Core i3-10110U @ 2.10 GHz, 4 logical cores |
| RAM | 7.86 GB total (free RAM varied from ~0.7 GB to ~1.2 GB) |
| GPU | None; `torch 2.10.0+cpu`, CUDA unavailable |
| OS | Windows 11 (`Windows-11-10.0.26200-SP0`) |
| Python | 3.12.5 |
| Ultralytics | 8.4.22 |
| Device setting | `cpu` |

The environment block is stored in every result JSON, so a rerun on different
hardware is not silently comparable.

---

## 3. Baseline facts (commit `8bbb3da`)

These values come from the checkpoint metadata, the dataset layout and the
pre-Stage-2 test run, as audited in
[`BASELINE_AUDIT.md`](BASELINE_AUDIT.md).

| Area | Baseline value | Provenance |
|---|---:|---|
| Vision precision | 0.87388 | `train_metrics`, best epoch in checkpoint |
| Vision recall | 0.69374 | same |
| Vision mAP@50 | 0.74229 | same |
| Vision mAP@50-95 | 0.40584 | same |
| Vision validation split | 10 images | `data.yaml` |
| Vision test split | 5 images | too small for statistical claims |
| Audio resubstitution accuracy | 1.0000 | measured on all 400 clips |
| Audio retained split / training script | none | not reproducible |
| Audio exact-duplicate pairs | 4 | inside `siren` |
| Automated tests | 16 passed | 40.53 s on this machine |
| Coverage | unmeasured | `pytest-cov` was absent |
| Latency / FPS instrumentation | none | no benchmark harness |

The baseline checkpoint stores a **best-epoch** snapshot. The final recorded
training epoch was materially worse (precision 0.98872, recall 0.50000,
mAP@50 0.59132), which already shows that 10 validation images give a noisy
model-selection signal.

---

## 4. Current vision measurement (test split, 640 px)

**Command**

```powershell
.\venv\Scripts\python.exe scripts\benchmark_vision.py --split test --imgsz 640 --no-plots
```

**Stored result:** `benchmarks/results/vision_test_640_20261110T084458Z.json`

| Metric | Value |
|---|---:|
| Precision (conf 0.001) | 0.9207 |
| Recall (conf 0.001) | 0.4000 |
| mAP@50 | 0.5217 |
| mAP@50-95 | 0.4088 |
| Preprocess | 3.85 ms/image |
| Inference | 2612.42 ms/image |
| Postprocess | 16.28 ms/image |
| Pipeline (pre + inf + post) | 2632.54 ms/image |
| Derived throughput | 0.3799 images/s |
| `val()` wall time | 20.234 s |
| Test images / labels | 5 / 5 |

**How to read it**

* The test split has **5 images**. The result file marks `small_sample: true`.
  These numbers are indicative only.
* mAP is threshold-independent and uses Ultralytics' default `conf=0.001`.
  Precision and recall above are at that evaluation threshold, not at the
  dashboard operating threshold of 0.65.
* The derived FPS is `1000 / (preprocess + inference + postprocess)` from the
  validation loop. It is not a live-camera guarantee.
* The same checkpoint produced **identical quality metrics** in the earlier
  stored run `benchmarks/results/yolo_test_640.json`
  (precision 0.9207, recall 0.4000, mAP@50 0.5217, mAP@50-95 0.4088). Speed
  differed between runs (3558 ms vs 2612 ms inference per image), which confirms
  that timing is load-sensitive while the quality values are stable for a fixed
  model, split and threshold.

**Baseline comparison:** not like-for-like.

| | Baseline | Current |
|---|---|---|
| Values | checkpoint `train_metrics` | fresh `val()` on test split |
| Split | 10-image validation | 5-image test |
| Model | same checkpoint | same checkpoint |

The current mAP@50 (0.5217) is lower than the baseline checkpoint value
(0.74229), but the split and evaluation context differ. This is **not** evidence
of a regression, and the higher precision is not evidence of an improvement.

---

## 5. Current audio measurement (all local clips)

**Command**

```powershell
.\venv\Scripts\python.exe scripts\benchmark_audio.py
```

**Stored result:** `benchmarks/results/audio_resubstitution_20261110T083904Z.json`

**Evaluation kind:** resubstitution on the 400 clips used for training.

| Metric | Value |
|---|---:|
| Clips | 400 (200 siren, 200 non-siren) |
| True positives / false negatives | 200 / 0 |
| True negatives / false positives | 200 / 0 |
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| Specificity | 1.0000 |
| F1 | 1.0000 |
| False-positive rate | 0.0000 |
| Decision threshold | 0.5 |

**Latency per clip**

| Stage | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|
| Load (`librosa.load`, 3.0 s window) | 52.68 ms | 17.65 ms | 28.90 ms | 13668.60 ms |
| MFCC extraction | 33.05 ms | 30.36 ms | 45.48 ms | 526.94 ms |
| Model prediction | 128.86 ms | 124.29 ms | 179.96 ms | 275.51 ms |
| End to end | 214.58 ms | 173.21 ms | 246.86 ms | 14372.10 ms |
| Throughput | 4.6599 clips/s over 85.84 s wall time | | | |

The single large load/end-to-end maximum is a cold-start/I/O spike. The median
and p95 are the useful steady-state figures; the mean is pulled upward by that
one sample. All four numbers are stored in the JSON.

**Baseline comparison:** the model and the 1.0000 resubstitution result are
unchanged. The baseline did not measure latency. The four exact-duplicate pairs
were detected by content hash and stored in the result JSON. Stage 11 now
excludes them from a deterministic split and adds a genuinely held-out
experiment: a new Random Forest trained on 277 files scored 1.0000 on 59
validation and 60 test files. The shipped classifier also scored 1.0000 on those
files, but its original training set is undocumented, so that is not a
validation result. The split is file-level held-out only; without source/session
metadata it is **not** source-independent. See
[`AUDIO_DATASET_PROVENANCE.md`](AUDIO_DATASET_PROVENANCE.md).

---

## 6. Current system measurement (simulated decision path)

**Command**

```powershell
.\venv\Scripts\python.exe scripts\benchmark_system.py
```

**Stored result:** `benchmarks/results/system_simulation_20261110T085500Z.json`

**Scenario:** 40 simulated seconds, 0.1 s decision interval, synthetic
video/audio evidence for Road 2 from 0–12 s, refreshed every 1.5 s. Fusion policy
`AUDIO`; signal timings from `config/default.yaml`.

### 6.1 Decision-path wall latency

Wall times are real `perf_counter` measurements of Python calls.

| Call | Samples | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|---:|
| Idle `tick()` | 400 | 0.1341 ms | 0.0962 ms | 0.2449 ms | 0.7114 ms |
| Emergency `tick()` | 400 | 0.1489 ms | 0.0972 ms | 0.3112 ms | 1.1690 ms |
| Transition `tick()` | 4 | 0.3185 ms | 0.2827 ms | 0.4596 ms | 0.4596 ms |
| `observe_video()` | 9 | 0.0392 ms | 0.0312 ms | 0.0725 ms | 0.0725 ms |
| `observe_audio()` | 9 | 0.0212 ms | 0.0163 ms | 0.0448 ms | 0.0448 ms |
| `snapshot()` | 200 | 0.0576 ms | 0.0388 ms | 0.0906 ms | 0.6657 ms |

Idle and emergency tick distributions overlap. A single run on this machine
does not support a claim that one is faster than the other.

### 6.2 Simulated timeline

These are `FakeClock` seconds, **not** wall time.

| Event | Simulated time |
|---|---:|
| First evidence | 0.0 s |
| Fusion reports active | 0.1 s |
| Green granted | 2.0 s |
| Activation delay | 2.0 s |
| Yellow starts | 17.1 s |
| Green duration | 15.1 s |
| Clearance red | 20.1 s |
| Normal red restored | 22.1 s |
| Yellow + all-red clearance | 5.0 s |

Four signal transitions were recorded: red→green, green→yellow, yellow→red and
red→normal. The 2.0 s activation delay is the configured 1.0 s confirmation
window overlapped with the configured 2.0 s all-red interval; it is a
configuration consequence, not a measured hardware delay.

### 6.3 Resources

| Scenario | Process RAM before → after | Delta |
|---|---:|---:|
| Idle | 203.156 → 203.191 MB | +0.035 MB |
| Emergency | 203.273 → 203.277 MB | +0.004 MB |

CPU samples were 91–100% during the idle run and 85.7% after the emergency run;
the first emergency sample was 0.0% because `psutil.cpu_percent(interval=None)`
compares against the previous call. CPU is therefore reported as observed, not
as a stable benchmark. The software decision path itself is sub-millisecond.

**Baseline comparison:** the baseline had no event-latency, transition-latency or
decision-cycle measurement at all. There is no baseline number to improve on.
The current values establish a reproducible floor for the decision path only;
full camera-to-signal latency still requires real video inference, which is
measured separately by the vision benchmark and the existing video-pipeline
harness.

---

## 7. Summary table

| Area | Baseline | Current | What changed |
|---|---|---|---|
| Vision quality | 0.8739 / 0.6937 / 0.7423 / 0.4058 | 0.9207 / 0.4000 / 0.5217 / 0.4088 | Evaluation protocol/split only; weights unchanged |
| Vision speed | not measured | 2632.54 ms/image, 0.3799 FPS | New measured harness |
| Audio quality | 1.0000 resubstitution | 1.0000 resubstitution (shipped) | No retraining of shipped model |
| Audio held-out (Stage 11) | not measured | 1.0000 on 60-file test split | Superseded; duplicate-contaminated file-level split |
| Audio content-group (Stage 12) | not measured | 1.0000 on 54-file test split | 234 groups held out; not source-independent |
| Audio latency | not measured | 173.21 ms median end-to-end | New measured harness |
| System decision path | not measured | 0.0972 ms median tick | New measured harness |
| System activation | not measured | 2.0 simulated seconds | Deterministic scenario result |
| Tests | 16 passed | 416 passed | 351 Stage 9 + 16 Stage 10 + 31 Stage 11–12 + 18 Stage 14 |
| Coverage | unmeasured | 92% statements in `src/ambulance_system` | Stage 9; statement coverage only |
| Audio duplicates | 4 pairs | 4 pairs excluded from the split | Detection + exclusion; quarantine not executed |
| Hardware boundary | modeled in code | simulation + mock controllers, 18 tests | No physical hardware opened or tested |

**Test-count provenance:** measured by `.\venv\Scripts\python.exe -m pytest -q`
on 2026-11-12: `416 passed, 1 warning in 46.05s`. The last coverage rerun with
`--cov=src/ambulance_system` passed 388 tests and reported 92% (221 missed of
2778 statements). No test was removed, skipped or weakened.

---

## 8. Reproduce everything

```powershell
# Vision: quality + validation-loop speed
.\venv\Scripts\python.exe scripts\benchmark_vision.py --split test --imgsz 640 --no-plots

# Audio: all 400 local clips, resubstitution only
.\venv\Scripts\python.exe scripts\benchmark_audio.py

# System: synthetic evidence, FakeClock, no models
.\venv\Scripts\python.exe scripts\benchmark_system.py

# Stage 11: audit and read-only content-group analysis
.\venv\Scripts\python.exe scripts\audio_integrity.py
.\venv\Scripts\python.exe scripts\audio_content_groups.py --report benchmarks\results\audio_content_groups_20261111.json

# Stage 12: content-group split and experiment-only retrain
.\venv\Scripts\python.exe scripts\make_audio_split.py --manifest data\audio\content_groups.csv --seed stage12-20261111
.\venv\Scripts\python.exe scripts\train_audio.py --split data\audio\splits\audio_split.json

# Full test suite
.\venv\Scripts\python.exe -m pytest -q

# Coverage (statement coverage)
.\venv\Scripts\python.exe -m pytest -q --cov=src/ambulance_system --cov-report=term-missing
```

Each script prints the JSON it wrote and stores the command, environment,
model hash (where applicable), configuration and timestamp inside the result.

---

## 9. What this comparison does **not** show

* It does not show that the vision model improved. The weights are unchanged.
* It does not show source-independent audio accuracy. Stage 11 adds a
  file-level held-out experiment (1.0000 on 60 test files), but the dataset has
  no session/source metadata, so windows from one recording may still span
  splits.
* It does not show real-time performance. Vision inference is ~2.6 s/image on
  this CPU; the system benchmark excludes inference entirely.
* It does not show traffic-control benefit. The signal timeline is a software
  simulation with synthetic evidence.
* It does not establish statistical significance. The test split has 5 images
  and each latency table is one run on one machine.

Real improvement claims require the held-out, source-independent audio split and
the licensed vision dataset expansion described in Stages 11 and 12.
