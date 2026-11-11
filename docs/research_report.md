# Final Research Report

## AI Ambulance Detection & Adaptive Traffic Signal Priority System

**Version:** `0.3.0`
**Date:** 2026-11-12
**Hardware:** Intel Core i3-10110U (4 logical cores), 7.9 GB RAM, CPU-only
**Software:** Python 3.12.5, Windows 11, PyTorch 2.10.0+cpu, Ultralytics 8.4.22
**Status:** research and engineering prototype; **not** a road-side deployment

---

## 1. Abstract

This report describes a modular, CPU-only prototype that combines visual
ambulance detection (YOLOv8 Large), acoustic siren classification (40 MFCCs +
Random Forest), rule-based multimodal fusion, and a simulated four-approach
traffic-signal controller. The work deliberately separates verified measurements
from experimental and blocked work. It also records a dataset-provenance
investigation that materially narrows the audio claims, and an offline synthetic
GCC-PHAT direction experiment that is explicitly disconnected from the priority
decision.

The defensible result is an engineering and evaluation platform: **443 automated
tests pass**, the source tree has **92% statement coverage**, and every reported
number is tied to a stored result or a reproducible command. The project does not
claim source-independent audio accuracy, improved vision weights, real-time
performance, traffic-control benefit, field localisation or physical hardware
validation.

---

## 2. Status of the work

| Label | Meaning |
|---|---|
| **Implemented** | Code exists and is used by the runtime or a CLI. |
| **Tested** | Covered by automated tests or a stored, reproducible measurement. |
| **Experimental** | Offline/synthetic work, not connected to the decision path. |
| **Planned** | Designed but not built. |
| **Blocked** | Cannot be completed with current hardware, data or licences. |

| Workstream | Status | Evidence |
|---|---|---|
| Configuration, events, clock, structured logging | Implemented, Tested | Stage 2 tests |
| Signal state machine v2 and arbiter | Implemented, Tested | Headless state/arbiter tests |
| YOLOv8 detection module and evaluation harness | Implemented, Tested | 5-image test split result |
| Video source abstraction (file/webcam/RTSP) | Implemented, Tested | Source/pipeline tests |
| Continuous audio streaming | Implemented, Tested | Streaming tests |
| Multimodal fusion engine | Implemented, Tested | Fusion/view-model tests |
| Dashboard upgrade and thread safety | Implemented, Tested | Headless view-model/widget tests |
| Test expansion and coverage | Implemented, Tested | 443 tests, 92% statements |
| Benchmark harness and baseline comparison | Implemented, Tested | Timestamped JSON results |
| Audio provenance and content-group remediation | Implemented, Tested | 396-file split, 234 groups |
| Vision dataset expansion and YOLO retraining | **Blocked** | Resource audit |
| GCC-PHAT direction research | **Experimental** | Synthetic 280-trial benchmark |
| Hardware abstraction (simulation/mock) | Implemented, Tested in software | 18 tests, no device |
| Documentation, limitations, release | Implemented | This report and companion docs |

The repository’s roadmap “Stage 12” is vision retraining. A separate Stage 12
audio follow-up (content-group split and experiment retrain) was completed in an
earlier session and is referred to below as the **audio Stage 12 follow-up** to
avoid ambiguity.

---

## 3. System overview

```text
Video file / webcam / RTSP ──► YOLOv8 detector ──┐
                                                 ├──► Fusion engine ──► Signal state machine ──► Simulated signal heads
Microphone / WAV file ──────► MFCC + RandomForest ┘         │                    │
                                                            │                    └──► HardwareController interface
                                                            │                         (simulation/mock only)
                                                            └──► Dashboard view-model ──► Tk dashboard

Offline, disconnected research:  two-channel audio ──► GCC-PHAT ──► far-field bearing
```

The runtime decision path does not import the localisation package or open a
hardware device. A source-scan test enforces the localisation separation; the
hardware package is likewise scanned for device-transport imports.

---

## 4. Verified results

### 4.1 Vision

**Dataset:** 117 images (102 train, 10 validation, 5 test), one class
(`ambulance`), CC BY 4.0, sourced from Roboflow Universe.

**Checkpoint metadata** (`yolov8l_ambulance.pt`, best epoch of 187):

| Metric | Value |
|---|---:|
| Precision | 0.87388 |
| Recall | 0.69374 |
| mAP@50 | 0.74229 |
| mAP@50-95 | 0.40584 |

These values are read from the checkpoint’s embedded `train_metrics`. The
original per-epoch artefacts were not retained, so they are not a fresh
reproduction. The final recorded epoch was worse (precision 0.98872, recall
0.50000, mAP@50 0.59132), which illustrates how noisy a 10-image validation split
is for model selection.

**Independent test-split harness** (5 images, `conf=0.001`, same checkpoint):

| Input | Precision | Recall | mAP@50 | mAP@50-95 | Inference |
|---:|---:|---:|---:|---:|---:|
| 640 px | 0.9207 | 0.4000 | 0.5217 | 0.4088 | 3558.3 ms/image |
| 320 px | 0.5714 | 0.8000 | 0.7217 | 0.4047 | 1611.7 ms/image |

Stage 10 re-measured the 640 px row and reproduced the quality values exactly
while measuring 2632.54 ms/image end-to-end (0.3799 images/s). Quality is stable
for a fixed model, split and threshold; speed is load-sensitive. **The checkpoint
was not retrained, so none of these numbers is an improvement claim.**

### 4.2 Audio

**Dataset:** 400 WAV files, nominally 200 siren + 200 non-siren. The provenance
audit found that this is not 400 independent recordings.

| Finding | Result |
|---|---|
| Files with a confirmed source and licence | 3 (`sound_85`–`87`, Freesound 87130 by dobroide, CC BY 4.0) |
| Probable excerpts from the same source | 3 (`sound_13`–`15`) |
| Files with no verified source or licence | 394 (397 not confirmed) |
| Non-siren independence | 200 files reduce to 60 unique 3 s clips, repeated 3–4 times |
| Siren duplicates | 4 byte-identical pairs, 3 further near-identical pairs, 32 related pairs at `r ≥ 0.80` |

**Stage 11 file-level split:** retained 396 files after excluding four duplicate
pairs; 277/59/60 train/validation/test. Its stored result
(`audio_training_20261111T070907Z.json`) is explicitly labelled
`held_out_file_level_duplicate_contaminated` and **must not be quoted as
held-out model quality**.

**Audio Stage 12 follow-up:** grouped the 396 retained files into 234 acoustic
content groups (174 siren, 60 non-siren) and generated a 280/62/54 split with
**0 groups spanning splits**:

| Split | Siren files | Non-siren files | Total | Siren groups | Non-siren groups |
|---|---:|---:|---:|---:|---:|
| train | 138 | 142 | 280 | 122 | 42 |
| val | 32 | 30 | 62 | 26 | 9 |
| test | 26 | 28 | 54 | 26 | 9 |

| Split | Samples | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 62 | 32 | 0 | 30 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 54 | 26 | 0 | 28 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Evaluation label: **`held_out_content_group`**. `source_independent` is
**false**. The shipped `siren_classifier.pkl` is unchanged (SHA-256
`2aa5a0dfd561dc0ba9d5a80873b65b39f979675079286c698a85c5e41f3a8e5c`) and also
scores 1.0000 on this split, but its original training set is undocumented.
Content groups are acoustic evidence, not verified sessions or sources.

**Measured classification latency** (all 400 local clips, resubstitution):
median 173.21 ms, p95 246.86 ms, from the stored Stage 10 result. This is not a
field latency measurement.

### 4.3 System decision path

The Stage 10 system harness feeds synthetic evidence into `DashboardViewModel`
with `FakeClock`; it excludes camera capture and model inference.

| Measurement | Result |
|---|---:|
| Median idle `tick()` | 0.0962 ms |
| Median emergency `tick()` | 0.0972 ms |
| Median snapshot | 0.0388 ms |
| Simulated activation delay | 2.0 s |
| Simulated yellow + all-red clearance | 5.0 s |
| Process RAM (idle → emergency) | ~203 MB |

The 2.0 s activation delay is the configured 1.0 s confirmation window
overlapping the 2.0 s all-red interval. It is a configuration consequence, not a
hardware delay. No traffic-control benefit was measured.

### 4.4 GCC-PHAT direction research (Experimental)

Synthetic free-field experiment: 16 kHz, 0.25 s windows, 0.12 m spacing, 16× IFFT
oversampling, 7 bearings from −75° to +75°, 10 trials per noise level.

| Sensor noise σ | Trials | Delay MAE | Bearing MAE | Within 2° |
|---:|---:|---:|---:|---:|
| 0.00 | 70 | 1.15 µs | 0.500° | 100% |
| 0.01 | 70 | 1.15 µs | 0.499° | 100% |
| 0.05 | 70 | 1.16 µs | 0.498° | 100% |
| 0.10 | 70 | 1.16 µs | 0.495° | 100% |
| **All** | **280** | **1.16 µs** | **0.498°** | **100%** |

Maximum observed bearing error was 1.26°. Two independent sources were detected
in 15/15 trials; the tested bearing pairs differed by roughly 4–6 native delay
samples, and the matched delay MAE was 2.00 µs. These are method checks under a
plane-wave model, **not** field accuracy. No microphone array was available, no
geographic position is computed, and the code is not wired into fusion or
priority.

Stored result:
`benchmarks/results/localisation_synthetic_20261111T191746Z.json`
(SHA-256 `93ec41ab86d2cb719e300718e3c622f363c8e339fdee4f4a31f236df1f0dc284`).

### 4.5 Test and coverage results

Measured on 2026-11-12 with `.\venv\Scripts\python.exe -m pytest -q`:

```text
443 passed, 1 warning in 40.20s
```

With `--cov=src/ambulance_system --cov-report=term-missing`:

```text
443 passed, 1 warning in 66.40s
TOTAL   3259 statements   255 missed   92%
```

Timing is load-sensitive: a later identical plain run took 63.05 s on the same
machine. The pass count, warning and coverage totals were unchanged.

The warning is the existing Librosa `audioread` deprecation notice. The 16
legacy tests remain unchanged. New Stage 13 tests: 27. Stage 14 tests: 18.
Stage 11–12 provenance tests: 31. No test was skipped or weakened.

---

## 5. Engineering contributions

1. **Modular architecture.** Configuration, events, clock, structured logging,
   detection, video sources, audio streaming, fusion, signal state machine,
   monitoring, dashboard, hardware boundary and localisation are separate
   packages with narrow interfaces.
2. **Deterministic, injectable evaluation.** Clock, capture, detector, classifier
   and RNG injection allow headless, offline tests and reproducible synthetic
   benchmarks.
3. **Signal safety model.** The state machine separates driver-visible colour from
   controller phase; the arbiter grants one GREEN at a time; the fault path
   forces all-red.
4. **Provenance-first data handling.** The audio audit detects duplicates,
   content groups and split leakage instead of reporting a misleading 1.0000.
5. **Traceable benchmarks.** Each result JSON stores the command, environment,
   configuration, model hash where applicable, and caveats.
6. **Explicit safety boundary.** The hardware package starts disconnected/all-red,
   forces red on connect/disconnect, and contains no device library. It is not
   wired into the dashboard.
7. **Honest status reporting.** Implemented, tested, experimental, planned and
   blocked work are labelled separately in this report and in
   [`limitations.md`](limitations.md).

---

## 6. Reproducibility

### 6.1 Commands

```powershell
# Full test suite
.\venv\Scripts\python.exe -m pytest -q

# Coverage
.\venv\Scripts\python.exe -m pytest -q --cov=src/ambulance_system --cov-report=term-missing

# Vision quality and speed (5-image test split)
.\venv\Scripts\python.exe scripts\benchmark_vision.py --split test --imgsz 640 --no-plots

# Audio resubstitution and latency
.\venv\Scripts\python.exe scripts\benchmark_audio.py

# Synthetic decision-path simulation
.\venv\Scripts\python.exe scripts\benchmark_system.py

# Audio audit, content groups, split and experiment retrain
.\venv\Scripts\python.exe scripts\audio_integrity.py
.\venv\Scripts\python.exe scripts\audio_content_groups.py --report benchmarks\results\audio_content_groups_20261111.json
.\venv\Scripts\python.exe scripts\make_audio_split.py --manifest data\audio\content_groups.csv --seed stage12-20261111
.\venv\Scripts\python.exe scripts\train_audio.py --split data\audio\splits\audio_split.json

# Synthetic localisation experiment
.\venv\Scripts\python.exe scripts\benchmark_localisation.py

# Dashboard launch
.\venv\Scripts\python.exe new.py
```

### 6.2 Stored hashes

| Artefact | SHA-256 |
|---|---|
| `siren_classifier.pkl` | `2aa5a0dfd561dc0ba9d5a80873b65b39f979675079286c698a85c5e41f3a8e5c` |
| Stage 12 audio split | `c878fa6a5100b0e2baf042387d1edf44c0e950053f4fcb8b8049c6a8716395da` |
| Stage 12 content-group CSV | `b94ae4c9159d527a274d8ee248ee85d4253262be2cceefd38547c69886d9871b` |
| Localisation synthetic result | `93ec41ab86d2cb719e300718e3c622f363c8e339fdee4f4a31f236df1f0dc284` |

The runtime loads only `yolov8l_ambulance.pt` and `siren_classifier.pkl`.

### 6.3 What cannot be reproduced from this repository

- YOLO per-epoch training curves and the original training split.
- The shipped siren classifier’s original training data and split.
- A source-independent audio evaluation (no verified session/source manifest).
- Field localisation or physical traffic-signal behaviour.

---

## 7. Release `0.3.0`

### Included

- Stages 2–11 and the audio Stage 12 follow-up.
- Stage 13 synthetic GCC-PHAT direction research (Experimental).
- Stage 14 simulation/mock hardware abstraction (no device access).
- Stage 15 documentation, limitations, research report and release notes.
- 443 passing tests; 92% statement coverage.

### Explicitly not included

- Vision Stage 12 retraining (blocked).
- Source-independent audio evaluation (blocked).
- Real microphone array, physical signal controller, live field trial
  (planned).
- Road-side safety certification (not attempted).

See [`release_notes.md`](release_notes.md) for the release checklist.

---

## 8. CV-ready bullets (verified work only)

These bullets are written to be defensible. Each one is followed by its evidence
and the caveat that must accompany it. Do not remove the caveats when quoting.

| Bullet | Evidence | Caveat |
|---|---|---|
| Built a modular Python 3.12 multimodal prototype combining YOLOv8 ambulance detection, MFCC + Random Forest siren classification, rule-based fusion and a simulated four-approach signal controller. | `src/ambulance_system/`, 443 passing tests | Software simulation; no physical signal was driven. |
| Developed an automated test and benchmark suite of 443 tests with 92% statement coverage, plus timestamped JSON harnesses that record environment, command, configuration and model hashes. | `pytest`, `benchmarks/results/` | Single CPU-only Windows machine; coverage is statement coverage, not correctness. |
| Audited a 400-clip audio dataset, found four byte-identical duplicate pairs and only three files with a confirmed licence, and replaced a duplicate-contaminated split with a 234-content-group split. | `docs/AUDIO_DATASET_PROVENANCE.md`, `data/audio/splits/audio_split.json` | The corrected result is content-group-held-out, **not** source-independent. |
| Trained and evaluated an experiment siren classifier on a 62-file validation and 54-file test content-group split, reporting 1.0000 accuracy without overwriting the shipped model. | `benchmarks/results/audio_training_20261111T181213Z.json` | Small, unverified-provenance collection; perfect score does not establish generalisation. |
| Measured a 5-image vision test-split evaluation at mAP@50 0.5217 (640 px) and 2632.54 ms/image on CPU, with the checkpoint unchanged. | `benchmarks/results/vision_test_640_20261110T084458Z.json` | Five images; not an improvement claim; CPU-only latency. |
| Designed a software-only hardware abstraction with all-red safe defaults, a simulation controller and a programmable mock, covered by 18 tests and a forbidden-import scan. | `src/ambulance_system/hardware/`, `tests/test_hardware.py` | No GPIO, relay, serial or physical hardware was tested. |
| Implemented an offline GCC-PHAT and far-field bearing experiment with known-delay synthetic tests; 280 trials gave 1.16 µs delay MAE and 0.498° bearing MAE. | `src/ambulance_system/localisation/`, `benchmarks/results/localisation_synthetic_20261111T191746Z.json` | Synthetic free-field only; no array, no geographic position, not wired into priority. |
| Documented limitations, non-claims and blocked work, separating Implemented / Tested / Experimental / Planned status. | `docs/limitations.md`, this report | Documentation does not create validation. |

**Do not use these bullets without the caveats:**

- “Deployed traffic-signal priority for ambulances.”
- “Achieved 100% source-independent siren accuracy.”
- “Localised ambulances geographically.”
- “Improved the YOLO model to 87% precision.”
- “Real-time ambulance detection at 0.38 FPS.”
- “Tested on real traffic hardware.”

None of those claims is supported by the evidence in this repository.

---

## 9. Future work

| Priority | Work | Status |
|---|---|---|
| 1 | Obtain a verified audio `session_id`/`source_id`/licence manifest and rerun a source-independent split | Blocked |
| 2 | Expand the vision dataset (night, rain, occlusion, distance, liveries) and retrain with a documented script | Blocked |
| 3 | Build a synchronised microphone array, calibrate it and collect field bearing ground truth | Planned |
| 4 | Add a physical signal adapter behind the Stage 14 interface with an independent emergency stop; bench-test with lamps | Planned |
| 5 | Run a controlled field trial with safety approval and travel-time baselines | Planned |
| 6 | Add CI for Windows/Linux and multiple Python versions | Planned |

---

## 10. Conclusion

The project demonstrates a complete, tested software pipeline for multimodal
ambulance detection and simulated signal priority, with unusually explicit
attention to data provenance, leakage control and non-claims. Its strongest
verified contributions are the modular architecture, the 443-test suite, the
traceable benchmark harness, the audio provenance investigation and the
content-group split. The vision and audio models remain small-data research
artefacts; the localisation and hardware layers are deliberately isolated
experiments. Road-side use would require substantially more data, hardware,
calibration, safety engineering and regulatory work.

---

## 11. Document map

| Document | Purpose |
|---|---|
| [`limitations.md`](limitations.md) | Full limitation and non-claim register |
| [`release_notes.md`](release_notes.md) | Release `0.3.0` contents and checklist |
| [`BASELINE_VS_IMPROVED.md`](BASELINE_VS_IMPROVED.md) | Baseline and current measurements |
| [`AUDIO_DATASET_PROVENANCE.md`](AUDIO_DATASET_PROVENANCE.md) | Audio provenance investigation |
| [`localisation.md`](localisation.md) | GCC-PHAT derivation and synthetic results |
| [`hardware_integration.md`](hardware_integration.md) | Simulation/mock hardware boundary |
| [`testing_report.md`](testing_report.md) | Per-stage test coverage |
| [`ADVANCED_DEVELOPMENT_ROADMAP.md`](ADVANCED_DEVELOPMENT_ROADMAP.md) | Stage definitions and status |
