# Baseline Audit — AI Ambulance Detection & Traffic Signal Priority System

**Audit date:** 2026-11-09
**Audited commit:** `8bbb3da3c08c1215ff46f62984e6097b6584c710` (`main` = `origin/main`)
**Method:** read-only inspection, metadata extraction, dataset integrity checks and one full test run.
**Source changes during this audit:** none. Only this document and the roadmap were added.

---

## 1. Environment at audit time

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU | Intel64 Family 6 Model 142, **4 logical cores** |
| RAM | **7.9 GB total, ~1.0 GB available** during audit |
| Disk | 237.6 GB total, **2.0 GB free** |
| Python | 3.12.5 |
| PyTorch | 2.10.0+cpu — **CUDA unavailable, 0 devices** |
| Ultralytics | 8.4.22 (checkpoint was trained with 8.3.226) |
| OpenCV | 4.13.0 |
| Git LFS | 3.7.1, active |

These constraints are material: they determine what can be trained or benchmarked locally.

---

## 2. Repository state

| Check | Result |
|---|---|
| Branch | `main`, tracking `origin/main`, working tree clean |
| Remote | `https://github.com/harshaba0408-netizen/ambulance-detection-project.git` |
| Commits | 1 published commit (`8bbb3da3c`); legacy 9-commit history preserved locally at `backup/full-history` (`770bb13af`) |
| Tracked files | 262 |
| LFS | `yolov8l_ambulance.pt` (LFS object, 87,657,530 bytes) |
| Secrets / personal paths | none found in tracked source |
| Unpublished-but-present data | 398 WAVs, 400 PNG spectrograms, 12 videos, duplicate checkpoint, `venv/` |

---

## 3. Current directory structure (source-relevant)

```
ambulance_detection_project/
├── new.py                                 # main 4-road Tkinter application (441 lines)
├── ambulance_and_siren_integration.py     # single-stream dual-modality script (182 lines)
├── ambulance_test.py                      # standalone YOLO video test (53 lines)
├── test_siren.py                          # standalone MFCC siren test (41 lines)
├── yolov8l_ambulance.pt                   # Git LFS, 87,657,530 bytes
├── Yovov8l_Amubalce_trained.pt            # local-only duplicate, 87,636,858 bytes
├── siren_classifier.pkl                   # RandomForestClassifier, 118,371 bytes
├── requirements.txt / pytest.ini / LICENSE / README.md
├── docs/
│   ├── architecture.md
│   ├── dataset_report.md
│   ├── model_evaluation.md
│   └── testing_report.md
├── tests/
│   ├── test_yolo_detection.py             # 5 tests
│   ├── test_audio_classification.py       # 5 tests
│   └── test_traffic_signal_logic.py       # 6 tests (uses a local mirror class)
└── data/
    ├── ambulance_images/                  # 117 images + 117 labels, data.yaml
    ├── audio/                             # 400 WAVs (2 published) + 400 PNGs (generated)
    └── Video/                             # 13 clips (1 published)
```

---

## 4. Source inventory

| File | Size / lines | Role | Import-safe? | Key observations |
|---|---:|---|---|---|
| `new.py` | 15,937 B / 441 | 4-approach dashboard, YOLO + RF + signal logic | Class import-safe; `tk.Tk()` only under `__main__` | Monolithic class; inference and UI coupled; detections latch permanently |
| `ambulance_and_siren_integration.py` | 5,373 B / 182 | Single-stream video + audio demo | **No** — loads models, opens video and runs loop at module import | Not unit-testable as written |
| `ambulance_test.py` | 1,724 B / 53 | CLI video detection demo | Yes (`run_ambulance_test`) | Uses `cvzone`; fixed class list |
| `test_siren.py` | 1,295 B / 41 | CLI siren classification demo | Yes (`predict_siren`) | File-based, 3 s window |
| `tests/test_yolo_detection.py` | 1,600 B / 50 | Model load + inference + error path | Yes | Real-image assertion uses **0.70** confidence; app uses **0.65** |
| `tests/test_audio_classification.py` | 2,236 B / 65 | Classifier load, MFCC shape, 2 predictions | Yes | No streaming/silence tests |
| `tests/test_traffic_signal_logic.py` | 3,725 B / 110 | 6 signal-logic tests | Yes | Tests a **duplicate mirror class**, not `new.py` — can drift |

**Supporting scripts:** `data/ambulance_images/python dataset_checks.py` (YOLO label validator, not wired into tests; resolves `data.yaml` paths relative to CWD, so it must be run from `data/ambulance_images/`), and two legacy `data/audio/*/sample.py` spectrogram generators (use `pylab.fromstring`, removed in modern Matplotlib; not referenced by runtime code).

---

## 5. Model artefacts and verified metadata

| Artefact | Bytes | SHA-256 |
|---|---:|---|
| `yolov8l_ambulance.pt` | 87,657,530 | `6784b16f90013517a55aec759710d7de1f87a468a8e0be9b9c3833588700229d` |
| `Yovov8l_Amubalce_trained.pt` | 87,636,858 | `d9353751967f015d936788d9b9716f24427689e4f5d7720ee83c30c815d90d80` |
| `siren_classifier.pkl` | 118,371 | `2aa5a0dfd561dc0ba9d5a80873b65b39f979675079286c698a85c5e41f3a8e5c` |

YOLO checkpoint (`yolov8l_ambulance.pt`, checkpoint date `2025-11-08T17:02:30`):

| Field | Value |
|---|---|
| Base model | `yolov8l.pt` (YOLOv8 Large), pretrained |
| Classes | `{0: 'ambulance'}` (`nc = 1`) |
| Input size | 640 × 640 |
| Epochs configured / recorded | 200 / **187** |
| Batch / patience / lr0 / optimizer | 16 / 100 / 0.01 / auto |
| Precision (B) | **0.87388 (87.39 %)** |
| Recall (B) | **0.69374 (69.37 %)** |
| mAP@50 (B) | **0.74229 (74.23 %)** |
| mAP@50-95 (B) | **0.40584 (40.58 %)** |
| val box / cls / dfl loss | 1.66486 / 1.46766 / 2.24910 |

Final epoch (187) values recorded in `train_results`: precision 0.98872, recall 0.50000, mAP@50 0.59132, mAP@50-95 0.25577. The headline table above is the **best-epoch** snapshot.

**Missing training artefacts:** `results.csv`, `best.pt`, `last.pt`, and any training script are absent. The metrics cannot be recomputed from this repository alone.

---

## 6. Vision dataset integrity (verified now)

`data/ambulance_images/data.yaml` → `nc: 1`, `names: ['ambulance']`, Roboflow project `ambulance-detection-idmmt` v1, **CC BY 4.0**.

| Split | Images | Labels | Boxes | Empty labels | Malformed / out-of-range | Classes |
|---|---:|---:|---:|---:|---:|---|
| train | 102 | 102 | 108 | 0 | 0 | `{0: 108}` |
| valid | 10 | 10 | 10 | 0 | 0 | `{0: 10}` |
| test | 5 | 5 | 5 | 0 | 0 | `{0: 5}` |
| **Total** | **117** | **117** | **123** | **0** | **0** | **1 class** |

* Exact duplicate images (MD5): **0**
* Duplicate groups spanning splits (leakage): **0**
* Class imbalance: single-class problem, so not applicable; box count per image is ~1.05.

The dataset is small but structurally clean. The 5-image test split is too small to support strong claims; it is a smoke test, not a benchmark.

---

## 7. Audio dataset and classifier (verified now)

| Item | Value |
|---|---|
| Files | 400 WAV — 200 `siren`, 200 `non-siren` |
| Duration | 3.00 s min/mean, 3.02 s max |
| Source sample rates | **44,100 Hz (341), 48,000 Hz (53), 22,050 Hz (6)** |
| Classifier | `RandomForestClassifier` — 100 estimators, `max_features='sqrt'`, `criterion='gini'`, `random_state=42`, 40 input features, 2 classes |
| Feature pipeline | `librosa.load(duration=3.0)` → `mfcc(n_mfcc=40)` → mean over time → shape `(40,)` |

**Findings that affect any future claim:**

1. **4 exact duplicate pairs exist inside the siren class:** `sound_129`=`sound_130`, `sound_73`=`sound_76`, `sound_74`=`sound_77`, `sound_75`=`sound_78`. There are also 3 further near-identical siren pairs (`1=34`, `2=35`, `3=36`) and 32 related pairs at `r >= 0.80` after alignment.
2. **The 200 non-siren files are not independent:** they reduce to 60 unique 3-second clips, each repeated 3–4 times in 60-file blocks. The Stage 11 provisional split therefore leaks content across train/val/test. Stage 12 regenerated the split with 60 non-siren and 174 siren acoustic content groups; no group spans two splits, but the groups are not verified sessions or sources. See `AUDIO_DATASET_PROVENANCE.md`.
3. **The original baseline retained no train/validation/test split and no training script.** Stage 10/11 added `make_audio_split.py` and `train_audio.py`; Stage 12 regenerated the split and retrained an experiment model. The shipped classifier is still not reproducible from the repository.
4. **Resubstitution accuracy is 100 %** on all 400 clips (200/200 each class). This is measured on the same data used to fit the model; it is **not** a validation or generalisation result. The Stage 11 file-level split also scored 1.0000 but is duplicate-contaminated. The Stage 12 content-group split scored 1.0000 val/test under the `held_out_content_group` label; that is leakage-controlled for known acoustic groups but still **not source-independent**.
5. Sample rates are mixed; inference relies on Librosa's implicit resampling to 22.05 kHz.
6. **Provenance is only partially recovered.** `sound_85`–`87` are verified excerpts of Freesound 87130 by dobroide (CC BY 4.0); `sound_13`–`15` are probable excerpts of the same recording. The other 394 files have no verified source or licence. `sound_88`–`90` carry a 2011-05-18 / Zoom H2n tag but remain unresolved.

The audio dataset is therefore a functional demo asset, not a validated benchmark.

---

## 8. Test baseline (executed)

Command: `.\venv\Scripts\python.exe -m pytest -q`

| Metric | Result |
|---|---|
| Collected | 16 |
| Passed | **16** |
| Failed | 0 |
| Skipped | 0 |
| Warnings | 1 (Librosa `__audioread_load` deprecation, triggered by the intentional invalid-path test) |
| Execution time | **40.53 s** (this run; earlier runs 26.89 s warm / 68.73 s cold) |

Test names:

* `test_yolo_detection.py` (5): model file, model loading, synthetic-frame inference, real-image inference, invalid path.
* `test_audio_classification.py` (5): classifier file, loading, MFCC dimension, siren/non-siren prediction, invalid audio path.
* `test_traffic_signal_logic.py` (6): default red, audio→green, video→green, both required, revert on exit, clear-all reset.

**Coverage is not measured** — `pytest-cov` is not installed.

**Correctness gap:** the 6 signal tests instantiate `TrafficSignalCoordinator`, a class defined *inside the test file*. They do not import or exercise `new.py`. A regression in the real application would not fail these tests.

---

## 9. Functional baseline

### Implemented and working (verified)
* YOLOv8-L checkpoint loads and performs inference (unit-tested).
* Ambulance detection on images/video frames with a 0.65 confidence gate.
* File-based MFCC + Random Forest siren classification.
* Tkinter dashboard with 4 approaches, per-road video/audio selection, policy dropdown (`AUDIO`, `VIDEO`, `BOTH`), status text, red/green lights, detection log, Clear All.
* Single-stream integration demo with visual/audio history windows and evidence-frame saving.
* 16 automated tests, all passing.
* Git LFS publication, clean README and four supporting documents.

### Not implemented
* Model training or fine-tuning scripts (vision or audio).
* Dataset expansion, augmentation pipeline, reproducible split generation.
* Continuous/streaming audio, microphone capture, confidence probabilities.
* Live RTSP or webcam abstraction; reconnection; FPS/latency instrumentation.
* Automatic priority release, hysteresis, yellow/clearing states, conflict arbitration, fail-safe/fault state.
* Temporal event model, duplicate suppression, stale-event expiry.
* Siren direction / angle-of-arrival.
* Hardware abstraction (simulation only).
* Structured logging, central configuration, code coverage, benchmarks.

### Known behavioural limits of the current app
* `video_detected` and `audio_detected` latch to `True` and are only cleared by **Clear All**.
* YOLO runs every 5th frame *until the first positive*, then never again for that road.
* Audio runs once per file selection, not continuously.
* No yellow phase, no minimum green, no transition timing.
* `self.log()` is called from worker threads and writes directly to a Tk widget (Tkinter is not thread-safe).
* Confidence threshold is duplicated in code (0.65) and tests (0.70).
* `ambulance_and_siren_integration.py` has module-level side effects, so it cannot be imported safely.

---

## 10. Gap analysis against the target capabilities

| Target capability | Current state | Gap severity |
|---|---|---|
| A. Better ambulance detection | Fixed 117-image model; recall 69.37 % | **High** — needs data + retraining (blocked by resources) |
| B. Continuous audio recognition | One-shot file classification | **High** |
| C. Live video/audio inference | File playback only | Medium |
| D. Adaptive signal priority | Latched booleans, no timing | **High** |
| E. Auto release + fail-safe | Manual Clear All only | **High** |
| F. Dashboard monitoring | Basic status/log; no FPS/latency/health | Medium |
| G. Testing + reproducibility | 16 tests; no coverage; models irreproducible | **High** |
| H. Hardware-ready architecture | Monolith, no interfaces | Medium |
| I. Performance reports | Baseline metrics only; no latency/FPS benchmarks | Medium |
| J. Controlled demo readiness | Works, but single-threaded one-shot logic | Medium |

---

## 11. Risks and technical debt

| Risk | Evidence | Impact |
|---|---|---|
| Audio leakage / duplication | 4 exact + 3 near-exact siren pairs; 200 non-siren files = 60 repeated clips; 44/60 groups crossed the Stage 11 split. Stage 12 holds out 234 acoustic groups (0 leaks) but cannot prove source independence | Inflated perceived accuracy; Stage 11 1.0000 invalid; Stage 12 1.0000 is content-group-held-out only |
| Unverifiable audio licence | Only `sound_85`–`87` traced (CC BY 4.0); 394 unresolved | **Do not publish the full audio set** |
| Non-reproducible models | No training scripts, no `results.csv` | Cannot defend metrics or improve systematically |
| Signal tests test a copy | Mirror class in test file | False confidence in signal logic |
| Permanent latching | Flags set once | Blocks auto-release requirement |
| Tk thread-safety | `log()` from workers | Intermittent UI crashes |
| Monolithic UI/business logic | 441-line class | Hard to test, refactor, extend |
| Threshold drift | 0.65 vs 0.70 | Inconsistent behaviour vs tests |
| Local resources | 2 GB free disk, 1 GB free RAM, no GPU | Training/benchmarks blocked locally |
| Checkpoint/version mismatch | Trained on 8.3.226, running 8.4.22 | Minor inference-compatibility risk |

---

## 12. Baseline preservation

* The published baseline commit is `8bbb3da3c08c1215ff46f62984e6097b6584c710`.
* The legacy 9-commit history is preserved locally as `backup/full-history` (`770bb13af`).
* Model weights **must not be overwritten**. Any retraining writes to `runs/experiments/<id>/weights/`.
* New experiments, datasets and configuration must not replace files under `data/ambulance_images/` or the published `.pt`/`.pkl` artefacts.

---

## 13. Audit conclusion

The baseline is a **working, honest software prototype**: detection and audio classification run, the dashboard is functional, the documentation matches measured metrics, and all 16 tests pass. Its weaknesses are reproducible science (no training code, no split), adaptive control (permanent latch, no timing), streaming (file-only audio), and engineering structure (monolith, no config/logging/metrics). Dataset expansion and YOLO retraining cannot be performed on this machine as configured.

Proceed to the staged plan in `docs/ADVANCED_DEVELOPMENT_ROADMAP.md`; Stage 2 is the only next step that requires no new data, no new hardware and no expensive compute.
