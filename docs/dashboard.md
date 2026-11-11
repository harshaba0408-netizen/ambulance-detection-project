# Dashboard Refactor and Thread Safety (Stage 8)

Stage 8 keeps the four-approach Tkinter dashboard but splits it into three
layers. The goal is a UI that stays responsive while inference runs, and whose
display logic can be tested without a screen, a model checkpoint or a video
file.

**Scope:** simulation only. The dashboard renders simulated signal states and
does not connect to physical traffic infrastructure.

---

## 1. Three layers

| Layer | Module | Responsibility | Imports Tkinter? |
|---|---|---|---|
| View-model | `src/ambulance_system/dashboard/viewmodel.py` | Owns fusion decisions, the Stage 3 `SignalController`, evidence state, metrics and the event history. Produces immutable snapshots. | No |
| Widgets | `src/ambulance_system/dashboard/widgets.py` | Draws `ApproachView` / `DashboardSnapshot` values. Never starts workers or inference. | Yes |
| Application | `src/ambulance_system/dashboard/app.py` | Builds the window, owns acquisition workers and the frame store, translates buttons into view-model commands, renders on the Tk main thread. | Yes |

`ambulance_system.dashboard.__init__` exports **only** the view-model types, so
importing the package does not import Tkinter. Tests import the view-model
directly.

`new.py` remains the launch command:

```bash
python new.py
```

It is now a thin entry point that calls `ambulance_system.dashboard.app.main`.
The configuration is loaded from `config/default.yaml` (or a path passed to
`main`), and the existing model files `yolov8l_ambulance.pt` and
`siren_classifier.pkl` are used unchanged.

---

## 2. Threading contract

```
Tk main thread                          Acquisition workers
──────────────                          ───────────────────
build widgets                           video pipeline on_frame()
render via root.after(100 ms)  ◄──────  observe_video() / update_video_stats()
button callbacks                        audio stream worker
tick() + snapshot()             ◄──────  observe_audio() / log_event()
```

* Every state change in `DashboardViewModel` is guarded by one re-entrant lock.
* Workers may call `observe_video`, `observe_audio`, `log_event`,
  `update_video_stats` and `update_health` from any thread.
* **Only the main thread** calls `tick()`, which evaluates fusion, advances the
  signal controller and returns the snapshot that is rendered.
* No worker thread touches a Tk widget. The previous `self.log()` calls from
  worker threads are gone; workers append to the view-model's event history
  instead, and the next render displays it.
* `refresh()` catches exceptions, records them as `ERROR` events, and always
  schedules the next `after()` callback, so one bad frame cannot freeze the UI.

Frames take a separate short-lived path: `FrameStore` keeps the latest resized
RGB frame and bounding box per approach under its own lock. `app.py` converts a
stored array to a `PhotoImage` on the main thread at render time only.

---

## 3. View-model API

```python
from ambulance_system.config import load_config
from ambulance_system.dashboard import DashboardViewModel

viewmodel = DashboardViewModel(load_config(), event_history=200)
viewmodel.start()

viewmodel.set_policy("Road 1", "VIDEO")          # AUDIO | VIDEO | BOTH | ADAPTIVE
viewmodel.observe_video("Road 1", 0.91, detected=True,
                        bbox=(12, 30, 140, 150), frame_index=42)
viewmodel.observe_audio("Road 1", 0.84, is_siren=True,
                        window_start_s=0.0, window_end_s=3.0)
viewmodel.update_video_stats("Road 1", fps=12.5, latency_ms=340.0,
                             frames_dropped=8, last_error=None)
viewmodel.update_health(cpu_percent=41.0, system_ram_percent=68.0,
                        process_ram_mb=512.0)

snapshot = viewmodel.tick()                      # main thread only
viewmodel.clear_all()                            # legacy Clear All
```

Key methods:

| Method | Thread | Purpose |
|---|---|---|
| `start()` / `stop()` | any | Set running state and log a lifecycle event. |
| `set_policy(approach, policy)` | main | Change one approach's policy; existing evidence is kept. Invalid policies or approaches raise `ValueError`. |
| `observe_video(...)` | worker | Record the latest video outcome and submit positive evidence. Returns whether fusion accepted a new event. |
| `observe_audio(...)` | worker | Record one classified window. A siren without a probability is stored as confidence `1.0`. |
| `tick(now=None)` | **main only** | Evaluate fusion, step the signal controller, append changed-state events, return a snapshot. |
| `snapshot()` | any | Return the current state without running a decision cycle. |
| `update_video_stats(...)` | worker/main | Store FPS, latency, dropped frames and the last video error. |
| `update_health(...)` | worker/main | Store measured CPU, system RAM and process RAM. |
| `log_event(level, message, approach=None)` | any | Append an `INFO` / `WARNING` / `ERROR` notification. |
| `clear_all(...)` | main | Legacy reset: stop being active, drop evidence, reset signals and video metrics, restore default policies, log the action. |
| `trigger_fault(...)` / `clear_fault(...)` | main | Force or acknowledge the simulated all-red `FAULT` state. |

Timestamps are monotonic seconds and must be finite and non-negative. The
signal controller rejects non-monotonic time, so callers should not run `tick`
with decreasing values.

### Snapshot types

All snapshot types are frozen dataclasses:

* `DashboardSnapshot` — timestamp, running flag, four `ApproachView` values,
  active approach, fault flag, `HealthView`, bounded `events` tuple, status
  message and level.
* `ApproachView` — name, policy, audio/video `ModalityView`, fused active flag,
  fused confidence and reason, signal phase/colour, priority start, signal
  reason, FPS, latency, dropped frames, video error, bounding box and frame
  index.
* `HealthView` — CPU, system RAM, process RAM, uptime and error count.
* `EventLogEntry` — monotonic sequence number, timestamp, level, message and
  optional approach.

Because snapshots are immutable, the UI can hold one while workers update the
view-model; a later `tick()` produces a new snapshot instead of mutating the one
being drawn.

---

## 4. What the dashboard shows

* **Four approach cards** — live preview, bounding box, policy selector,
  audio/video state and confidence, fused state and reason, current signal
  colour and phase, FPS, latency and dropped frames.
* **Signal head** — red / yellow / green lamps driven by the controller's
  current colour.
* **System health panel** — active approach, fault flag, measured CPU, system
  RAM, process RAM, uptime and error count.
* **Event history** — timestamped, colour-coded `INFO` / `WARNING` / `ERROR`
  messages, capped at 300 lines on screen and 200 entries in the view-model.
* **Controls** — per approach: Select Video, Select Audio, Start, Stop and a
  policy menu; globally: Clear All, Clear Fault and Simulate Fault.

`Clear All` still performs the legacy reset: it stops sources, clears stored
files and preview frames, drops all evidence, returns every signal to red and
restores the default policy. It is intentionally a manual action; normal
release is handled by fusion staleness and the controller's hysteresis.

---

## 5. Per-approach policies

Each approach has its own policy, initialised from `fusion.policy` in
`config/default.yaml`. Changing one card's policy does not affect the others.

`app.py` starts the sources a policy needs:

| Policy | Video source | Audio source |
|---|---|---|
| `AUDIO` | stopped | started when a file is selected |
| `VIDEO` | started when a file is selected | stopped |
| `BOTH` | started | started |
| `ADAPTIVE` | started | started |

If `BOTH` or `ADAPTIVE` is selected without the required file, the dashboard
records a `WARNING` instead of silently pretending the approach is monitored.

The policy rules themselves — confidence handling, staleness, debounce and the
tighter ADAPTIVE window — are documented in [`fusion.md`](fusion.md). The
dashboard does not re-implement them.

---

## 6. Tests

`tests/test_viewmodel.py` adds 27 headless tests (18 test functions, one
parameterised with 10 invalid-input cases). They import no Tkinter and use
`FakeClock`, so no display, model or video file is required. Coverage includes:

* initial all-red state and per-approach policy overrides;
* policy changes keeping existing evidence;
* video, audio, `BOTH` and `ADAPTIVE` confirmation paths;
* duplicate suppression, staleness expiry and bounding-box clearing;
* the one-green invariant, signal lifecycle and legacy `Clear All`;
* fault forcing and explicit recovery;
* metrics, bounded event history, error counting and snapshot immutability;
* concurrent logging and evidence submission from six worker threads;
* rejection of unknown approaches, bad policies, out-of-range confidences,
  negative metrics and non-finite timestamps.

Stage 8 also adds two fusion-engine tests for per-approach policies and
`reset()`. Stage 9 adds five application-shell tests (`FrameStore` plus the app
with injected fakes) and five widget-rendering tests; `tests/conftest.py` keeps
one hidden Tk root per session. The full suite is run with:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

A manual UI smoke test is kept outside the repository: it builds the real
window against a withdrawn Tk root, injects a fake detector and classifier, and
closes through `app.on_close`. Automated tests do not open a visible window;
the Tk rendering tests skip with an explicit reason only when no display is
available. The latest full result (351 tests, 92% statement coverage) is in
[`testing_report.md`](testing_report.md).

---

## 7. Limitations

* The dashboard is a **software simulation**. It does not drive physical
  signals and makes no traffic-safety claim.
* Video is file playback; webcam and RTSP sources exist in the video module but
  are not validated on hardware or a live stream in this dashboard.
* Audio is file-backed streaming. Live microphone capture remains blocked on
  the optional `sounddevice` dependency.
* Rendering is intentionally simple Tkinter; heavy inference can still reduce
  preview frame rate, although it no longer blocks the event loop or corrupts
  widget state.
* FPS and latency values are measured from the actual run and are not
  real-time guarantees for any particular machine.
* Selecting a different file for a modality that is already running takes
  effect after **Stop** and **Start** (or a policy change); the running worker
  is not replaced in place.
