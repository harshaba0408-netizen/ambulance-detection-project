# Model Evaluation & Benchmark Metrics
## AI Ambulance Detection & Traffic Signal Priority System

---

## 1. Computer Vision: YOLOv8 Large (`yolov8l_ambulance.pt`)

The visual detection pipeline utilizes a fine-tuned **YOLOv8 Large** deep neural network trained on emergency vehicle imagery.

### Training Configuration
- **Base Architecture:** YOLOv8 Large (`yolov8l.pt`)
- **Number of Parameters:** ~43.7M parameters
- **Input Resolution:** 640 × 640 pixels
- **Batch Size:** 16
- **Training Epochs:** 187 (Early stopped out of 200 with patience = 100)
- **Optimizer:** Auto (SGD/AdamW with warmup)
- **Initial Learning Rate:** 0.01

### Verified Validation Performance (Embedded Checkpoint Metadata)
*These are the **best-epoch** values stored in the checkpoint's `train_metrics` dictionary. The original `results.csv`, `best.pt`, and `last.pt` training artefacts are not present in this repository, so per-epoch curves cannot be reconstructed from the repo alone.*

| Evaluation Metric | Measured Value | Percentage |
| :--- | :--- | :--- |
| **Precision (B)** | `0.87388` | **87.39%** |
| **Recall (B)** | `0.69374` | **69.37%** |
| **mAP@50 (B)** | `0.74229` | **74.23%** |
| **mAP@50-95 (B)** | `0.40584` | **40.58%** |
| **Validation Box Loss** | `1.66486` | — |
| **Validation Class Loss** | `1.46766` | — |
| **Validation DFL Loss** | `2.24910` | — |

For transparency, the final epoch (187) recorded in `train_results` was: Precision `0.98872`, Recall `0.50000`, mAP@50 `0.59132`, mAP@50-95 `0.25577`. The best-epoch values above are the ones reported throughout this project.

### Comparison Across Checkpoint Iterations

| Checkpoint Name | Date | Epochs | Precision | Recall | mAP@50 | mAP@50-95 | Selected for Production |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Yovov8l_Amubalce_trained.pt` | 2025-11-08 (10:34) | Initial | 97.76% | 60.00% | 74.46% | 35.80% | No (Lower Recall) |
| `yolov8l_ambulance.pt` | 2025-11-08 (17:02) | 187 | **87.39%** | **69.37%** | **74.23%** | **40.58%** | **YES (Balanced & higher mAP50-95)** |

> `Yovov8l_Amubalce_trained.pt` is an earlier comparison checkpoint. It is **not published** in the repository (excluded by `.gitignore`) and is not loaded by any script; only `yolov8l_ambulance.pt` is used at runtime.

### Independent test-split evaluation (Stage 4 harness)

Command:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_yolo.py --split test --imgsz 640 --no-plots --output benchmarks\results\yolo_test_640.json
```

Measured on the published checkpoint (`sha256 6784b16f…0229d`), CPU-only `torch 2.10.0+cpu`, Intel Core i3-10110U (4 logical cores), 7.9 GB RAM. Evaluation used `conf=0.001` (the mAP protocol), **not** the dashboard's operating threshold of 0.65. The test split contains **5 images with 5 labelled instances**.

| Input size | Precision | Recall | mAP@50 | mAP@50-95 | Inference (ms/image) |
|---:|---:|---:|---:|---:|---:|
| 640 × 640 | 0.9207 | 0.4000 | 0.5217 | 0.4088 | 3558.3 |
| 320 × 320 | 0.5714 | 0.8000 | 0.7217 | 0.4047 | 1611.7 |

Raw results: `benchmarks/results/yolo_test_640.json` and `benchmarks/results/yolo_test_320.json`.

Stage 10 re-measured the same checkpoint and split with `scripts/benchmark_vision.py`:

```powershell
.\venv\Scripts\python.exe scripts\benchmark_vision.py --split test --imgsz 640 --no-plots
```

The quality metrics were identical to the 640 row above (precision 0.9207, recall 0.4000, mAP@50 0.5217, mAP@50-95 0.4088), while inference time differed (2612.4 ms/image; 2632.54 ms/image including preprocess and postprocess; 0.3799 images/s). The stored result is `benchmarks/results/vision_test_640_20261110T084458Z.json`. Speed varies with machine load; quality does not, for a fixed checkpoint, split and threshold. Full comparison: [`BASELINE_VS_IMPROVED.md`](BASELINE_VS_IMPROVED.md).

**How to read this table**

* It is an independent held-out test split, but five images cannot support a generalisation claim. The two rows differ dramatically in recall because a single miss changes recall by 20 percentage points.
* These values are **not** comparable to the 10-image validation metrics above, and they are not evidence of improvement or regression. They are a reproducible check that the harness runs and records real output.
* Inference latency is measured on this CPU only and includes Ultralytics validation overhead per image. It is not a real-time FPS claim.
* The first 640 × 640 attempt with plots enabled exceeded 15 minutes on this machine and produced no output; `--no-plots` was required to complete in ~18 s. This is an environment limitation, not a model property.

---

## 2. Audio Processing: Siren Acoustic Classifier (`siren_classifier.pkl`)

### Feature Representation
- **Feature Set:** Mel-Frequency Cepstral Coefficients (MFCC)
- **Number of Coefficients:** $40$
- **Sampling Rate:** $22,050\text{ Hz}$
- **Analysis Window:** $3.0\text{ seconds}$
- **Feature Vector:** $1 \times 40$ dimensional mean feature vector

### Model Architecture
- **Classifier:** `sklearn.ensemble.RandomForestClassifier(random_state=42)`
- **Task:** Binary Classification
  - `0`: Non-Siren (Ambient traffic, car horns, engines, city noise)
  - `1`: Siren (Emergency ambulance sirens)
- **Dataset Size:** 400 audio recordings (200 Siren, 200 Non-Siren)
- **Classes Balance:** Exactly 50/50 balance across training sets

