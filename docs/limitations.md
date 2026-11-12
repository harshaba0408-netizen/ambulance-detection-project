# Limitations, Status and Non-Claims

**Project:** AI Ambulance Detection & Traffic Signal Priority System
**Document status:** final for release `0.3.0`
**Last updated:** 2026-11-12
**Companion documents:** [`research_report.md`](research_report.md),
[`BASELINE_VS_IMPROVED.md`](BASELINE_VS_IMPROVED.md),
[`AUDIO_DATASET_PROVENANCE.md`](AUDIO_DATASET_PROVENANCE.md)

This document states what is implemented, what was actually tested, what is
experimental, what is only planned, and what the project must **not** be claimed
to do. Numbers are quoted only where a stored result or a measured test run
supports them.

---

## 0. Status labels used in this repository

| Label | Meaning |
|---|---|
| **Implemented** | Code exists in the repository and is exercised by the runtime or a CLI. |
| **Tested** | Covered by automated tests or a stored, reproducible measurement on this machine. |
| **Experimental** | Offline or synthetic work that is not connected to the runtime decision path and is not validated in the field. |
| **Planned** | Designed or listed in the roadmap but not built. |
| **Blocked** | Cannot be completed with the current hardware, data or licence constraints. |

“Tested” never means field-tested. No traffic signal, microphone array, live
camera or road-side deployment was available.

---

## 1. Status summary

| Capability | Status | Evidence | Principal limitation |
|---|---|---|---|
| YOLOv8 ambulance detector | Implemented, Tested | Checkpoint metadata; 5-image test-split evaluation | Five test images; no retraining; CPU-only |
| MFCC + Random Forest siren classifier | Implemented, Tested | 396-file content-group split; 1.0000 val/test | Not source-independent; single channel; no direction |
| Video source abstraction (file/webcam/RTSP) | Implemented, Tested | Source and pipeline tests | No live RTSP or camera validation |
| Audio streaming | Implemented, Tested | Streaming tests; latency benchmark; 31 backend-free live-source tests | Live accuracy unvalidated; no intersection deployment |
| Multimodal fusion policies | Implemented, Tested | Fusion and view-model tests | Rule-based thresholds, not learned or field-tuned |
| Signal state machine and arbiter | Implemented, Tested | Headless state-machine and arbiter tests | Software simulation; no physical signal |
| Dashboard | Implemented, Tested | Headless view-model and widget tests; manual launch | CPU-only; no long-run soak test |
| Benchmark harness | Implemented, Tested | Timestamped JSON with environment and commands | One machine; synthetic or resubstitution audio |
| Audio provenance tooling | Implemented, Tested | Audit, content groups, split and provenance tests | 394 files have no verified source or licence |
| Hardware abstraction | Implemented, Tested | Simulation/mock controllers, 18 tests | No GPIO, relay or serial device; not wired in |
| GCC-PHAT direction research | Experimental | Synthetic benchmark, 27 tests | No microphone array; free-field model only |
| Vision dataset expansion / retraining | Blocked | Resource audit | Disk, GPU, RAM, licensed data and training script absent |
| Source-independent audio evaluation | Blocked | Provenance audit | No verified source/session manifest |
| Road-side deployment | Planned | — | No safety case, hardware, field data or approval |

---

## 2. Vision limitations

1. **The dataset is small.** `data/ambulance_images/` contains 117 images split
   102 train / 10 validation / 5 test. The test split has 5 labelled instances.
   A single miss changes recall by 20 percentage points.
2. **The reported checkpoint metrics are not a fresh reproduction.** Precision
   0.87388, recall 0.69374, mAP@50 0.74229 and mAP@50-95 0.40584 come from the
   best-epoch `train_metrics` embedded in `yolov8l_ambulance.pt`. The original
   `results.csv`, `best.pt` and `last.pt` files were not retained, so the
   per-epoch training curve cannot be reconstructed from this repository.
3. **The independent test result is indicative only.** On the 5-image test split
   at `conf=0.001`, the harness measured precision 0.9207, recall 0.4000,
   mAP@50 0.5217 and mAP@50-95 0.4088 at 640 px. These values are not
   comparable to the 10-image validation `train_metrics`, and they are not
   evidence of improvement or regression because the checkpoint was not
   retrained.
4. **Coverage is narrow.** The images do not span night, rain, fog, heavy
   occlusion, unusual ambulance liveries or a representative range of camera
   viewpoints. No claim about those conditions is supported.
5. **Single class.** The detector predicts only `ambulance`; it does not identify
   direction of travel, lane, distance or emergency status.
6. **CPU inference is slow.** The measured pipeline was 2632.54 ms per image
   (0.3799 images/s) on this CPU. A live multi-camera deployment would need
   hardware acceleration and a different latency budget.
7. **`Yovov8l_Amubalce_trained.pt` is not used.** Only
   `yolov8l_ambulance.pt` is loaded at runtime. The comparison checkpoint is kept
   locally and is not published.
8. **Retraining is blocked.** This machine has about 2 GB free disk, no GPU,
   4 CPU cores and limited free RAM; no licensed additional dataset and no
   training script are present. No retraining result is claimed.

---

## 3. Audio limitations

1. **Provenance is unverified.** Only `sound_85`–`87` are traced to a named,
   licensed source: Freesound 87130, `20091229.ambulance.siren.wav` by
   **dobroide**, CC BY 4.0. `sound_13`–`15` are probable excerpts from the same
   recording. The remaining **394 files have no verified source or licence**;
   397 files in total are not confirmed.
2. **The 400 files are not 400 independent recordings.** The 200 non-siren files
   reduce to **60 unique 3-second clips**, each repeated 3–4 times. The siren set
   contains four byte-identical pairs (`73=76`, `74=77`, `75=78`, `129=130`),
   three further near-identical pairs, and 32 related pairs at `r ≥ 0.80` after
   alignment.
3. **The old 1.0000 scores are not held-out evidence.** The Stage 11 stored
   file-level result (`benchmarks/results/audio_training_20261111T070907Z.json`)
   is **duplicate-contaminated** and must not be quoted as model quality.
4. **The corrected result is content-group-held-out, not source-independent.**
   Stage 12 retains 396 files, forms 234 acoustic content groups (174 siren,
   60 non-siren), and generates a 280/62/54 train/val/test split with **0 groups
   spanning splits**. The experiment model and the shipped classifier both score
   1.0000 on the 62-file validation and 54-file test splits under the label
   `held_out_content_group`. Content groups are acoustic evidence, not verified
   sessions or sources.
5. **Perfect scores do not establish generalisation.** Different excerpts of one
   source can remain separate groups; the collection is small, balanced by
   construction, and not a random sample of real traffic noise. Performance on
   new sirens, recorders, cities or weather is unknown.
6. **The shipped model was not replaced.** `siren_classifier.pkl` is unchanged
   (SHA-256 `2aa5a0dfd561dc0ba9d5a80873b65b39f979675079286c698a85c5e41f3a8e5c`)
   and its original training set remains undocumented. Its 1.0000 score on the
   Stage 12 split is not a validation result for that model.
7. **Single microphone, no direction.** The classifier answers “siren present or
   absent”. It cannot estimate bearing, distance, lane or which road an
   ambulance is on. `docs/localisation.md` describes a separate synthetic
   experiment that also does not provide a geographic position.
8. **Feature and threshold choices are fixed.** The pipeline uses 40 MFCCs, a
   3.0 s window, 22.05 kHz loading and a 0.5 decision threshold. These were not
   optimised on a verified held-out field set.
9. **Latency is not real-time.** Median end-to-end classification was 173.21 ms
   with p95 246.86 ms per 3 s clip on this CPU. Load, MFCC and prediction costs
   will differ on other hardware.
10. **Redistribution is unresolved.** The audio must not be redistributed until
    its licences are confirmed, even though only two fixtures are published in
    this repository.
11. **Live capture is unvalidated and mis-calibrated on near-silence.**
    `MicrophoneAudioSource` and `scripts/stream_microphone.py` exist, but on
    this machine the microphone intermittently returned uninitialised buffers
    (RMS ~1e24) and near-silent audio (RMS ~1.5e-5); one near-silent 3 s window
    was scored as SIREN with confidence 0.87–0.93. No recorded siren was played
    through the device, so the live path has no accuracy figure. Invalid blocks
    are rejected by the capture layer, but the classifier's energy calibration
    remains an open data/model issue.

---

## 4. Fusion and signal-control limitations

1. **Policies are hand-written rules.** `AUDIO`, `VIDEO`, `BOTH` and `ADAPTIVE`
   use confidence thresholds, staleness windows and debounce timers. They are
   not learned from traffic data and were not tuned at a real intersection.
2. **The signal controller is a software simulation.** The state machine,
   arbiter, phase timing and all-red fault path are implemented and tested in
   software. No physical signal head was driven.
3. **“Activation delay” is a configuration consequence.** The measured 2.0 s
   delay is the configured 1.0 s confirmation window overlapping the 2.0 s
   all-red interval. It is not a hardware or road-side latency measurement.
4. **The decision-path benchmark is synthetic.** It feeds fixed-confidence events
   into the view-model with `FakeClock`. The median emergency `tick()` cost of
   0.0972 ms excludes camera capture, YOLO inference and audio classification.
5. **No traffic-benefit claim.** No vehicle was given priority, no travel time was
   measured, and no congestion or safety outcome was evaluated.
6. **Single-approach priority.** The arbiter grants GREEN to one approach at a
   time and holds others at red. Conflicting or simultaneous emergency vehicles,
   pedestrians, pre-emption recovery and manual override were not field-tested.
7. **One run, one machine.** The latency tables are single runs on a CPU-only
   Windows machine. They do not establish statistical significance or
   cross-platform performance.

---

## 5. Dashboard and runtime limitations

1. **The dashboard is a software simulator.** It visualises video, audio,
   evidence and signal state; it does not control a real traffic light.
2. **CPU-only preview.** Frame rate depends on the machine and on how many
   approaches are running YOLO inference.
3. **UI tests are mostly headless.** View-model and widget tests run without a
   display where possible; the Tk application shell has lower coverage. A manual
   launch was verified on this machine, but there is no long-duration soak test,
   multi-user test or accessibility review.
4. **No installation service.** The runtime is launched from a virtual
   environment with `python new.py`; there is no packaged installer, container,
   watchdog or auto-restart.
5. **No live source validation.** File-based video has been exercised; live
   webcam and RTSP sources are implemented and unit-tested with fakes, not
   validated against real network cameras.

---

## 6. GCC-PHAT localisation (Experimental)

1. **Synthetic only.** The 280-trial single-source check and 15-trial two-source
   check use generated white noise and a free-field plane-wave model. No
   microphone array, siren recording or real room was used.
2. **Accuracy is a method check, not a field result.** Under the synthetic
   model, delay MAE was 1.16 µs, bearing MAE 0.498°, maximum bearing error 1.26°,
   and 100% of trials were within 2°. These numbers do not transfer to a real
   array without calibration and field data.
3. **No geographic position.** The output is a far-field bearing relative to an
   assumed geometry. A position requires multiple baselines or another method
   and was not implemented.
4. **Free-field assumptions.** Reverberation, walls, diffraction, sensor
   directivity, gain mismatch, clock drift, wind and temperature are not
   modelled. A nearby siren violates the plane-wave assumption.
5. **Ambiguity and two-source limits.** A single pair and a collinear array
   cannot resolve front/back or full 2-D bearing. GCC-PHAT returns peaks, not
   source identities; closely spaced or unequal sources can merge or hide.
6. **Not wired into priority.** A source-scan test proves the localisation
   package imports no fusion, signal, dashboard or device module. No decision
   depends on it, and no siren was tracked.
7. **Geometry bound.** For the 0.12 m synthetic pair, the largest unambiguous
   delay is 349.9 µs (5.60 samples at 16 kHz). That is a geometry property, not
   an accuracy claim.

---

## 7. Hardware boundary (Implemented, Tested in software only)

1. **No physical hardware was opened or tested.** The package provides a
   `HardwareController` protocol, a default in-memory `SimulationController` and
   a `MockController` test double. It imports no GPIO, relay, serial or network
   library.
2. **Not wired into the dashboard.** The runtime still displays a software
   signal simulation. No green aspect can be energised by this code.
3. **Safety invariants are software invariants.** Starting disconnected and
   all-red, forcing red on connect/disconnect, and rejecting writes while
   disconnected are tested behaviours, not certified electrical fail-safes.
4. **A physical prototype is future work.** It would require isolated
   low-voltage lab hardware, direct human supervision, an independent emergency
   shutdown, bench testing with lamps or LEDs, and a written safety case. None
   of that exists.

---

## 8. Testing and reproducibility limitations

1. **Test count and coverage.** The full suite is **474 passed, 1 warning** on
   2026-11-12 (Python 3.12.5, CPU-only Windows, 86.16 s). The **92% statement
   coverage** (255 missed of 3259 statements, 66.40 s) was measured on the
   earlier 443-test suite and was not re-measured after the 31 Stage 6
   live-microphone tests were added. Coverage measures executed statements,
   not correctness or field readiness.
2. **The warning is real.** The Librosa `audioread` deprecation warning is
   triggered by the legacy invalid-path test. It is not suppressed.
3. **One machine only.** All timings come from one CPU-only Windows machine.
   There is no CI matrix, Linux/macOS run, GPU run or statistical repetition.
4. **No hardware-in-the-loop.** No GPIO, relay, Arduino, Raspberry Pi, ESP32 or
   traffic-signal controller was connected.
5. **No live-camera or live-microphone soak test.** The dashboard launch was a
   manual smoke test, not a durability test.
6. **Some modules have limited coverage.** `dashboard/app.py` is at 68% because
   model loading, file dialogs, worker threads and `main()` are not unit-tested.
   The localisation simulation module is at 86% and covers only the synthetic
   paths exercised by tests.
7. **Training artefacts are incomplete.** Per-epoch YOLO curves cannot be
   reproduced, and the shipped audio classifier's original training set is
   undocumented. The Stage 12 experiment model is stored under ignored
   `artifacts/` and is not the shipped model.
8. **The suite is not a safety certification.** Passing tests do not establish
   fitness for road-side use.

---

## 9. Data licensing and redistribution

| Asset | Published state | Limitation |
|---|---|---|
| Vision images | 117 images, CC BY 4.0, attributed to Roboflow Universe | Exact source dataset terms should be rechecked before redistribution |
| Audio | 2 fixtures published; 398 WAVs kept local | Only 3 files have a confirmed licence; 394 have no verified source/licence |
| Video | 1 sample (`data/Video/1.mp4`) published | The local stock-footage previews have no documented redistribution licence |
| YOLO weights | `yolov8l_ambulance.pt` via Git LFS | Training data and artefact provenance are only partially documented |
| Siren classifier | `siren_classifier.pkl` | Original training data and split are undocumented |

Do not redistribute the full audio or video collections, or describe the project
as having a clear licence for them, until the missing provenance is resolved.

---

## 10. Deployment and safety limitations

1. **Not a road-side system.** The project is a research and engineering
   prototype. It has no regulatory approval, safety certification, redundancy,
   watchdog, fail-safe relay or independent emergency stop.
2. **No cybersecurity review.** File dialogs, RTSP URLs, configuration files and
   model loading were not assessed for hostile input in a deployed environment.
3. **No privacy review.** Traffic video and audio can contain personal data.
   Retention, consent, anonymisation and jurisdiction-specific rules were not
   addressed.
4. **No accessibility or human-factors study.** The dashboard is an engineering
   interface, not an operator console validated for stress or accessibility.
5. **Do not connect this code to a live signal.** A physical adapter would need
   isolated hardware, human supervision, a manual emergency shutdown and a
   formal safety case. The Stage 14 package must not be treated as that adapter.

---

## 11. Claims this project does not make

- It does **not** claim source-independent audio accuracy.
- It does **not** claim that the vision model improved; the weights are unchanged.
- It does **not** claim real-time performance; vision inference is about
  2.6 s/image on this CPU.
- It does **not** claim traffic-control benefit, reduced congestion or improved
  emergency response times.
- It does **not** claim direction, distance or geographic position from the
  single-microphone siren classifier.
- It does **not** claim field localisation accuracy from the synthetic
  GCC-PHAT experiment.
- It does **not** claim that any physical traffic signal, GPIO pin, relay or
  serial device was tested.
- It does **not** claim road-side safety, certification or production readiness.
- It does **not** claim that 400 audio files are 400 independent recordings.

---

## 12. Blocked and planned work

| Item | Status | What is missing |
|---|---|---|
| Vision dataset expansion and YOLO retraining | **Blocked** | Free disk, GPU, RAM, licensed diverse data, training script |
| Source-independent audio evaluation | **Blocked** | Verified `session_id`/`source_id` and licence manifest for 394 files |
| Real microphone-array localisation | **Planned** | Synchronised multi-channel hardware, calibration, field recordings |
| Physical signal adapter | **Planned** | Isolated hardware, bench tests, emergency shutdown, safety case |
| Live RTSP / camera soak test | **Planned** | Network cameras and a long-running test environment |
| Field traffic trial | **Planned** | Site approval, safety review, baseline travel-time data |
| CI matrix (OS / Python versions) | **Planned** | Repository CI configuration and runners |

---

## 13. Bottom line

The defensible description is: a **modular, tested, CPU-only research
prototype** that combines a small fine-tuned YOLOv8 detector, an MFCC + Random
Forest siren classifier, a rule-based multimodal fusion layer and a simulated
four-approach signal controller, with reproducible benchmarks, explicit data
provenance tooling and an offline synthetic direction experiment.

It is **not** a deployed traffic-control system, a source-independent audio
benchmark, a geographic localisation system or a validated road-safety product.
