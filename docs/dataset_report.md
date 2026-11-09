# Dataset Specification Report
## AI Ambulance Detection & Traffic Signal Priority System

---

## 1. Vision Dataset (`data/ambulance_images/`)

- **Source:** Roboflow Universe (`yolov8-ambulance-detection/ambulance-detection-idmmt/1`)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Annotation Format:** YOLO standard text files (`class_id center_x center_y width height` normalized between 0.0 and 1.0)
- **Classes:** 1 (`0: ambulance`)

### Split Breakdown

| Dataset Split | Number of Images | Number of Labels | Pair Ratio |
| :--- | :--- | :--- | :--- |
| **Train** | 102 | 102 | 100% |
| **Validation** | 10 | 10 | 100% |
| **Test** | 5 | 5 | 100% |
| **Total** | **117** | **117** | **100% Verified** |

### Preprocessing & Data Augmentation Applied (at export)
1. Auto-orientation of pixel data (stripping EXIF tags).
2. Random brightness variation ($\pm 10\%$).
3. Random exposure adjustment ($\pm 12\%$).

---

## 2. Acoustic Dataset (`data/audio/`)

- **Total Audio Files:** 400 WAV recordings
- **Audio Format:** WAV format, PCM encoded, 44.1 kHz stereo source; loaded at 22.05 kHz by default in `librosa.load()`
- **Label Mapping:**
  - `data/audio/siren/`: 200 ambulance siren samples (`sound_1.wav` to `sound_200.wav`)
  - `data/audio/non-siren/`: 200 non-siren environmental samples (`sound_401.wav` to `sound_600.wav`)
- **Balance:** 1:1 ratio between emergency siren sounds and urban traffic noises.
- **Provenance note:** the audio clips do not carry an explicit licence file in this repository. Their original source/licence should be confirmed before public redistribution; the vision dataset is the only component with a documented CC BY 4.0 licence.

---

## 3. Video Test Footage (`data/Video/`)

The local working copy contains 13 benchmark video and audio clips:
- Numbered traffic test pairs (`1.mp4`, `1.wav`, `2.mp4`, `2.wav`, `3.mp4`, `3.wav`)
- Full HD traffic streams (`126687260_main_xxl.mp4`, `277926641_main_xxl.mp4`)
- High-density Indian traffic clips (Chennai and Uttar Pradesh street footage featuring ambulances navigating congestion).

**Published subset:** only `data/Video/1.mp4` (633 KB) is committed, as a runnable sample for `ambulance_and_siren_integration.py`. The remaining clips stay local because of size and because the `stock-footage-*` previews have no documented redistribution licence.

