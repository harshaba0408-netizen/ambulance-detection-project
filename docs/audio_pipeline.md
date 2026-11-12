# Offline and Live Audio Streaming

Stage 6 replaced the one-shot, first-three-seconds audio check with a sliding
window over the whole file and added a live microphone source on top of the same
windowing code. The dashboard (`new.py`) still analyses selected audio files;
live capture is available through `scripts/stream_microphone.py`.

Live capture is **not** a validated field result. `sounddevice` is optional at
runtime (installed in this environment as 0.5.6), the device is opened at the
configured rate without resampling, and the classifier was not trained for live
ambient noise. Measurements and the observed failure modes are in section 7.

This stage makes no improvement claim for the siren classifier. The published
checkpoint and its provenance caveats are documented in
`docs/BASELINE_AUDIT.md` and `docs/model_evaluation.md`; no retraining or data
cleaning happened here.

---

## 1. Components

```
src/ambulance_system/audio/features.py    load_audio + one MFCC implementation
src/ambulance_system/audio/classifier.py  SirenClassifier wrapper + probabilities
src/ambulance_system/audio/devices.py     FileAudioSource + MicrophoneAudioSource
src/ambulance_system/audio/streaming.py   sliding windows, predictions, summary
scripts/stream_microphone.py              live capture CLI (Stage 6 follow-up)
```

### Feature parity

`mfcc_features` is the single implementation of the legacy feature step:

```python
y, sr = librosa.load(path, duration=3.0)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
feat = np.mean(mfccs.T, axis=0)
```

A test compares the new function against that exact calculation on a real siren
sample (`tests/test_audio_features.py::test_mfcc_matches_the_legacy_feature_pipeline`).
`load_audio` uses librosa's 22050 Hz mono defaults, so the dashboard no longer
depends on whatever sample rate a file happens to use.

### Classifier wrapper

`SirenClassifier` keeps the hard-label behaviour (`predict(...) == 1`) and adds:

* the positive-class probability from `predict_proba` when available,
* a configurable threshold (`audio.threshold`, default 0.5),
* input validation for feature vectors and sample windows.

For the published RandomForest the threshold and the estimator's own argmax
agree at 0.5. If an estimator has no `predict_proba`, `confidence` is `None` and
the hard label is used. When converting a window to a `DetectionEvent`, a
missing probability is recorded as 0.0/1.0 rather than invented.

### Sources

`FileAudioSource` decodes the file once (the published clips are ~3 s) and then
returns fixed-size mono blocks (`audio.block_s`, default 0.25 s).

`MicrophoneAudioSource` implements the same `AudioSource` protocol from a live
input device:

* `sounddevice` is imported lazily in `open()`/`list_input_devices()`, so the
  package still imports and its tests run when the optional backend is absent.
* The PortAudio callback hands blocks to a bounded queue (`buffer_blocks`,
  default 8 → 2 s). If the consumer falls behind, the newest block is dropped
  and counted in `dropped_blocks`.
* Empty, non-finite or out-of-range blocks (peak > `MAX_ABS_SAMPLE = 4.0`) are
  rejected and counted in `invalid_blocks`. A blocked Windows microphone has
  been observed returning uninitialised buffers (RMS around 1e24), which would
  otherwise crash feature extraction.
* Two-channel input is downmixed to mono by averaging. The device is opened at
  `audio.sample_rate`; there is no resampling layer, so an unsupported rate
  raises `AudioSourceError`.
* `read()` waits up to `read_timeout_s` (default 2 s) and then raises
  `AudioSourceError` with the rejected-block count. `close()` stops the stream
  and unblocks a reader.

`list_input_devices()` returns `AudioDeviceInfo` records for input-capable
devices only, hiding output endpoints.

## 2. Sliding-window behaviour

With `window_s = 3.0`, `overlap_s = 1.5`:

```
window 0: [0.0, 3.0)          full
window 1: [1.5, 4.5)          full or partial at end-of-stream
window 2: [3.0, 6.0)          ...
```

`hop = window_s - overlap_s = 1.5 s`. Windows are classified independently.
The final window may be shorter than `window_s`; it is marked `partial=True`
and uses only the samples that exist. There is **no zero-padding**, because
padding would change the features relative to the legacy path.

For the published 3.0-second clips this yields two windows per file: `[0, 3.0]`
full and `[1.5, 3.0]` partial. The dashboard therefore examines the tail that
the old `duration=3.0` load already covered, plus the final milliseconds of
files that are slightly longer than three seconds.

## 3. Aggregation and events

`summarize(predictions)` returns `windows`, `siren_windows`, `max_confidence`
and `detected` (`siren_windows > 0`). `new.py` uses `detected` to set the same
per-road siren flag as before.

`WindowPrediction.to_event(approach)` produces the shared audio event:

```python
DetectionEvent.audio(
    approach="Road 1",
    confidence=0.93,
    window_start_s=1.5,
    window_end_s=3.0,
)
```

This is the evidence object that Stage 7 fusion will consume; it is not yet
connected to the signal state machine.

## 4. Integration with `new.py`

* `load_models()` now builds a `SirenClassifier` from
  `config/model.siren_classifier_path`, threshold and sample rate.
* `start_audio_detection()` runs `analyze_file(path, classifier, CONFIG.audio)`
  in a worker thread and logs window counts and maximum confidence.
* The existing `result_queue` / `audio_detected` latching is unchanged.
* `pickle`, `librosa` and `numpy` imports were removed from `new.py`; loading
  and feature extraction now live in the package.

Behaviour change: the whole file is scanned instead of only the first 3
seconds. No file in the repository is long enough for this to add many windows,
and the result flag is still a single boolean.

## 5. Measured checks (this machine)

Command used (with `src` on `PYTHONPATH`):

```powershell
.\venv\Scripts\python.exe -c "from ambulance_system.audio import ...; ..."
```

After a one-file warm-up (librosa/Numba initialisation takes ~8 s on the first
call), two 3.0-second files were analysed on CPU:

| File | Windows | Siren windows | Max confidence | Wall time |
|---|---:|---:|---:|---:|
| `data/audio/siren/sound_1.wav` | 2 | 2 | 1.000 | 0.256 s |
| `data/audio/non-siren/sound_401.wav` | 2 | 0 | 0.000 | 0.315 s |

These are timing observations for two files, not a classifier performance
result. The 1.000/0.000 values are this RandomForest's output on the training
distribution. The Stage 11 provenance investigation found that the audio set is
much more duplicated than the original four siren pairs suggested: the 200
non-siren files reduce to 60 unique 3-second clips repeated 3-4 times, and 44 of
those 60 groups cross the stored train/val/test boundaries. The stored 1.0000
held-out scores are therefore duplicate-contaminated and must not be quoted as
model quality. See [`AUDIO_DATASET_PROVENANCE.md`](AUDIO_DATASET_PROVENANCE.md).

Stage 12 regenerated the split with **234 acoustic content groups** (60
non-siren, 174 siren) and retrained. No content group spans two splits, so the
result is labelled **`held_out_content_group`**. The experiment scored 1.0000 on
validation (62 files) and test (54 files), but this is **not** a
source-independent result: content groups are acoustic evidence, not verified
sessions or sources. See section 9.1 of
[`AUDIO_DATASET_PROVENANCE.md`](AUDIO_DATASET_PROVENANCE.md).

**Cold-start note:** the first audio analysis in a fresh process takes about
8 seconds because librosa initialises its Numba kernels. Later files take a few
hundred milliseconds. The dashboard does not yet warm up the pipeline at start.

## 6. Tests

Stage 6 added 31 tests in the original offline pass; the live-source follow-up
adds 31 more, plus 8 silence-gate tests in `tests/test_audio_streaming.py`
(70 audio-stage tests, 482 in the full suite):

| File | Coverage |
|---|---|
| `tests/test_audio_features.py` | legacy MFCC parity, resampling/downmixing, determinism, input validation |
| `tests/test_audio_classifier.py` | real checkpoint, threshold policy, probability fallback, validation |
| `tests/test_audio_streaming.py` | window geometry, overlap, partial tails, block source, event conversion, real siren/non-siren files, silence-gate skip/allow/validation |
| `tests/test_audio_microphone.py` | fake `sounddevice` backend: lifecycle, mono downmix, timeout, close/unblock, buffer overflow, invalid blocks, device listing, shared-pipeline integration |

The 16 legacy audio/vision tests remain unchanged and passing.

## 7. Live microphone capture (2026-11-12)

```powershell
# list input-capable devices and exit
.\venv\Scripts\python.exe scripts\stream_microphone.py --list-devices

# capture until Ctrl+C, or for a bounded run
.\venv\Scripts\python.exe scripts\stream_microphone.py --device 1 --seconds 20

# adjust or disable the silence gate
.\venv\Scripts\python.exe scripts\stream_microphone.py --min-rms 0.01
.\venv\Scripts\python.exe scripts\stream_microphone.py --min-rms 0
```

The CLI loads the normal config and checkpoint, warms the first MFCC call
**before** opening the device, then prints one line per window. The warm-up is
deliberate: the first librosa/Numba call takes ~8 s on this machine, and without
it the bounded queue overflows and about 5 s of live audio is lost (measured: 21
dropped blocks in one run; 0 after the fix).

Measured on this machine (Windows, Realtek microphone, 22050 Hz, mono):

| Check | Result |
|---|---|
| `--list-devices` | 7 input endpoints reported; output endpoints hidden |
| Device opened at 22050 Hz, 0.25 s blocks | works; 5512-sample callbacks |
| `--seconds 1` (3 s window) | 1 window classified; 0 dropped blocks after warm-up |
| Device health | intermittent uninitialised buffers (RMS ~1e24) and noise-floor captures (RMS ~1.5e-5) |
| `--seconds 10`, gate off (before) | 5 windows, **5 siren**, confidence 0.87–0.89 on a muted input |
| `--seconds 10`, default gate (after) | 5 windows, **0 siren**, 5 reported `silent` |

**Important negative result:** the microphone on this machine was not delivering
trustworthy audio. Some captures returned garbage buffers, others sat at the
noise floor, and a near-silent 3-second window was classified **SIREN with
confidence 0.87–0.93**. That is a classifier calibration failure on
out-of-distribution, near-silent input — not evidence of a siren. Therefore:

* the capture layer is unit-tested with an injected fake backend, so no
  hardware is required in CI;
* the real device loop was exercised, but **live siren accuracy is not
  validated** and no live detection metric is quoted;
* live output is indicative only and must not be connected to a signal
  controller.

The energy/calibration gap belongs to the classifier and data work
(Stages 11–12), not to the capture layer. See
[`limitations.md`](limitations.md).

### Silence gate

`stream_predictions` takes `min_rms` (default `0.0` = disabled). When set, a
window whose RMS is below the floor is **not sent to the classifier** and is
returned as `is_siren=False, confidence=None`; the CLI prints it as `silent` and
counts it separately in the summary. The gate lives in the shared streaming
function (not in the classifier) and is off by default, so offline file analysis
and the dashboard are unchanged.

```python
for prediction in stream_predictions(source, classifier,
                                     window_s=3.0, overlap_s=1.5,
                                     min_rms=1e-3):
    ...
```

The CLI default is `--min-rms 1e-3`, roughly two orders of magnitude above the
measured noise floor (~1.5e-5) and well below normal speech/music energy. It is
a **live-input guard, not a validity check**: it removes the muted-microphone
false positive measured here, but it cannot make the classifier's accuracy on
real siren audio known. `--min-rms 0` restores the previous behaviour.

## 8. Still blocked

* **Live microphone validation** needs a known-good microphone with Windows
  privacy access granted and a recorded siren source; neither was available
  here. The device code exists, but no live accuracy figure is claimed.
* **Audio retraining or data cleaning** still needs verified provenance, a
  retained train/test split and a training script. None exist in the repo.
* The classifier pickle was written with scikit-learn 1.6.1 and is loaded under
  1.8.0; the version warning is suppressed at load time. Results are accepted as
  baseline behaviour, not as a reproducibility guarantee.
