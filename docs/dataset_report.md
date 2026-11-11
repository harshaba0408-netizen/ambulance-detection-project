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
- **Provenance note:** the audio clips do not carry an explicit licence file in this repository. Only `sound_85`–`87` have been traced to a named source: Freesound 87130 (`20091229.ambulance.siren.wav` by **dobroide**, recorded 2009 in Seville, **CC BY 4.0**). `sound_88`–`90` carry a 2011-05-18 / Zoom H2n recorder tag but their source is unresolved. The remaining 394 files have no verified source or licence; only `sound_88`–`90` carry additional recorder tags (2011-05-18 / Zoom H2n), and their source is unresolved.
- **Independence warning:** the 200 non-siren files are **not 200 independent clips**. Content analysis found **60 unique 3-second clips**, each repeated 3-4 times in 60-file blocks (20 groups of 4, 40 groups of 3). The siren set contains four byte-identical pairs (`73=76`, `74=77`, `75=78`, `129=130`), three further near-identical pairs, and 32 related pairs at `r >= 0.80` after alignment. The stored Stage 11 1.0000 audio scores are therefore **duplicate-contaminated**, not valid held-out estimates. Stage 12 regenerated the split with **234 acoustic content groups** (60 non-siren, 174 siren; 0 groups span splits) and retrained; the new result is labelled **`held_out_content_group`** and also scored 1.0000, but it is **not source-independent**. See [`AUDIO_DATASET_PROVENANCE.md`](AUDIO_DATASET_PROVENANCE.md) for the full evidence, split-leakage counts and recommended next action.

---

## 3. Video Test Footage (`data/Video/`)

The local working copy contains 13 benchmark video and audio clips:
- Numbered traffic test pairs (`1.mp4`, `1.wav`, `2.mp4`, `2.wav`, `3.mp4`, `3.wav`)
- Full HD traffic streams (`126687260_main_xxl.mp4`, `277926641_main_xxl.mp4`)
- High-density Indian traffic clips (Chennai and Uttar Pradesh street footage featuring ambulances navigating congestion).

**Published subset:** only `data/Video/1.mp4` (633 KB) is committed, as a runnable sample for `ambulance_and_siren_integration.py`. The remaining clips stay local because of size and because the `stock-footage-*` previews have no documented redistribution licence.

