# System Architecture & Technical Workflows
## AI Ambulance Detection & Traffic Signal Priority System

---

## 1. System Architecture Overview

The system operates as a multimodal, dual-channel edge AI pipeline designed to decongest urban intersections during emergency vehicle transit.

```mermaid
graph TD
    subgraph SENSORS ["Multimodal Input Streams"]
        A1["Traffic Surveillance Cameras (RTSP/MP4)"]
        A2["Acoustic Sensors / Microphones (WAV/PCM)"]
    end

    subgraph CV_PIPELINE ["Vision Pipeline (YOLOv8 Large)"]
        B1["Frame Acquisition & Preprocessing (640x640)"]
        B2["YOLOv8l Feature Extraction & CSPDarknet Backbone"]
        B3["Bounding Box & Confidence Scoring (Threshold >= 0.65)"]
        B4["Visual Detection State"]
    end

    subgraph AUDIO_PIPELINE ["Audio Pipeline (MFCC + Random Forest)"]
        C1["Audio Chunk Acquisition (3.0s window)"]
        C2["Librosa MFCC Feature Extraction (n_mfcc=40)"]
        C3["Random Forest Classifier (100 Trees, Binary)"]
        C4["Acoustic Siren State"]
    end

    subgraph COORDINATOR ["Intelligent Signal Decision Engine"]
        D1["Fusion Policy Dispatcher (AUDIO / VIDEO / BOTH / ADAPTIVE)"]
        D2["Evidence Staleness, Hysteresis & Conflict Arbiter"]
        D3["Traffic Light State Machine (Road 1 - 4)"]
    end

    subgraph OUTPUTS ["System Actions & Dashboards"]
        E1["Traffic Signal Lights (Red -> Green Transition)"]
        E2["Interactive 4-Road Tkinter Dashboard"]
        E3["Evidence Logging & Timestamped Frame Capture"]
    end

    A1 --> B1 --> B2 --> B3 --> B4
    A2 --> C1 --> C2 --> C3 --> C4
    B4 --> D1
    C4 --> D1
    D1 --> D2 --> D3
    D3 --> E1
    D3 --> E2
    D2 --> E3
```

---

## 2. YOLOv8 Ambulance Detection Workflow

```mermaid
sequenceDiagram
    autonumber
    participant Camera as Video Feed
    participant OpenCV as Frame Preprocessor
    participant YOLO as YOLOv8l Model
    participant Engine as Priority Controller

    Camera->>OpenCV: Read video frame (BGR)
    OpenCV->>OpenCV: Resize & RGB conversion (640x640)
    OpenCV->>YOLO: Pass tensor into YOLOv8l
    YOLO->>YOLO: Run single-class inference (0: ambulance)
    YOLO-->>OpenCV: Return [x1, y1, x2, y2, conf, cls]
    alt conf >= 0.65 and cls == 0
        OpenCV->>Engine: Signal Ambulance Detected (Video = TRUE)
    else conf < 0.65 or no detections
        OpenCV->>Engine: Signal Clear (Video = FALSE)
    end
```

---

## 3. Acoustic Siren Processing Workflow

```mermaid
flowchart LR
    A["Raw Audio Stream (.wav)"] --> B["Librosa Resampling (22.05 kHz)"]
    B --> C["Framing & Short-Time Fourier Transform (STFT)"]
    C --> D["Mel Filterbank Application (n_mels=128)"]
    D --> E["Discrete Cosine Transform (DCT)"]
    E --> F["40 MFCC Coefficients (Mean across time)"]
    F --> G["Random Forest Classifier"]
    G --> H{"Class == 1?"}
    H -->|Yes| I["Siren Confirmed (Audio = TRUE)"]
    H -->|No| J["Background / Traffic (Audio = FALSE)"]
```

---

## 4. Multi-Road Signal Priority Logic

The decision engine coordinates traffic states across an intersection with 4 approaches (**Road 1** through **Road 4**):

```mermaid
stateDiagram-v2
    [*] --> RedState: Initialized

    state RedState {
        [*] --> RedLightActive: Normal Cycle
        RedLightActive --> CheckPolicy: Input Triggered
    }

    state GreenPriorityState {
        [*] --> GreenLightActive: Emergency Mode
        GreenLightActive --> PriorityHold: Maintain Green
    }

    CheckPolicy --> GreenPriorityState: Detection Matches Policy (AUDIO / VIDEO / BOTH)
    GreenPriorityState --> RedState: Vehicle Clears / Siren Stops / Manual Clear
```

- **AUDIO Policy:** Requires positive siren acoustic classification.
- **VIDEO Policy:** Requires visual ambulance bounding box above threshold ($\ge 0.65$).
- **BOTH Policy:** Requires fresh evidence from both modalities within the configured staleness window, reducing false alarms.
- **ADAPTIVE Policy:** `BOTH` plus a tighter temporal window; it is never looser than `BOTH`.

> **Implemented behaviour note (Stage 8):** `new.py` is now a thin launcher for `ambulance_system.dashboard.app`. The dashboard's headless view-model owns a `FusionEngine` and the Stage 3 `SignalController`; audio and video evidence expires after `fusion.stale_after_s`, the controller applies activation/release hysteresis and transition timing, and only one approach can hold GREEN. Detection flags are no longer latched: priority is released when evidence goes stale or the release rules are met. **Clear All** remains a manual full reset (sources, evidence, metrics, signals and policies), and **Simulate Fault** forces all-red until **Clear Fault**. See [`dashboard.md`](dashboard.md), [`fusion.md`](fusion.md) and [`signal_state_machine.md`](signal_state_machine.md).

## 5. Hardware Boundary (Stage 14)

The signal controller is separated from any physical output by `ambulance_system.hardware`:

```text
FusionEngine → SignalController → HardwareController (interface)
                                   ├── SimulationController  (default, in-memory)
                                   └── MockController        (tests only)
```

* The interface exposes `connect`, `disconnect`, `apply`, `apply_all_red` and `snapshot`.
* `SimulationController` records timestamped commands and never opens a device.
* Every controller starts disconnected and all-red; a colour is only accepted while connected.
* The hardware layer does not validate phase transitions; that stays in `traffic_signal`.
* It is **not** wired into the dashboard, and no GPIO, relay, serial or network device is imported.

A physical adapter is future work and must keep the same safe defaults plus an independent emergency shutdown. See [`hardware_integration.md`](hardware_integration.md).

## 6. Offline Direction Research (Stage 13, Experimental)

`ambulance_system.localisation` is a separate, offline branch:

```text
two synchronised channels → GCC-PHAT TDOA → far-field bearing
```

* `gcc_phat.py` estimates the delay between two channels with a phase-transform
  cross-correlation.
* `doa.py` converts pair delays into a bearing, or solves a 2-D bearing from two
  or more non-collinear pairs.
* `simulation.py` generates synthetic plane-wave and two-source signals with
  known delays for tests and the benchmark.

The branch is **not imported by fusion, the signal controller or the dashboard**.
It does not identify sirens, does not assign an approach and does not produce a
geographic position. Its only published results are synthetic free-field method
checks. See [`localisation.md`](localisation.md).

