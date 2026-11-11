# Offline Audio Streaming

Stage 6 replaces the one-shot, first-three-seconds audio check with a sliding
window over the whole file. **Only the offline file path is implemented.** Live
microphone capture is deferred until `sounddevice` is installed and approved;
`new.py` continues to analyse selected audio files.

This stage makes no improvement claim for the siren classifier. The published
checkpoint and its provenance caveats are documented in
`docs/BASELINE_AUDIT.md` and `docs/model_evaluation.md`; no retraining or data
cleaning happened here.

---

## 1. Components

```
src/ambulance_system/audio/features.py    load_audio + one MFCC implementation
src/ambulance_system/audio/classifier.py  SirenClassifier wrapper + probabilities
src/ambulance_system/audio/devices.py     FileAudioSource (mono block source)
src/ambulance_system/audio/streaming.py   sliding windows, predictions, summary
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
returns fixed-size mono blocks (`audio.block_s`, default 0.25 s). A future
microphone source only has to implement the same `AudioSource` protocol. No
microphone code, device enumeration or PortAudio dependency exists yet.

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

Stage 6 adds 31 tests:

| File | Coverage |
|---|---|
| `tests/test_audio_features.py` | legacy MFCC parity, resampling/downmixing, determinism, input validation |
| `tests/test_audio_classifier.py` | real checkpoint, threshold policy, probability fallback, validation |
| `tests/test_audio_streaming.py` | window geometry, overlap, partial tails, block source, event conversion, real siren/non-siren files |

The 16 legacy audio/vision tests remain unchanged and passing.

## 7. Still blocked

* **Live microphone capture** needs `sounddevice` (or `pyaudio`) installed; it
  is not present. No device code was added.
* **Audio retraining or data cleaning** still needs verified provenance, a
  retained train/test split and a training script. None exist in the repo.
* The classifier pickle was written with scikit-learn 1.6.1 and is loaded under
  1.8.0; the version warning is suppressed at load time. Results are accepted as
  baseline behaviour, not as a reproducibility guarantee.
