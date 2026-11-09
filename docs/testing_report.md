# Automated Testing & Quality Assurance Report
## AI Ambulance Detection & Traffic Signal Priority System

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

