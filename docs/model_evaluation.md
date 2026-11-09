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

