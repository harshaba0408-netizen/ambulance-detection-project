from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ambulance_system.audio import (
    AudioWindow,
    FileAudioSource,
    SirenClassifier,
    WindowPrediction,
    analyze_file,
    iter_windows,
    stream_predictions,
    summarize,
)
from ambulance_system.config import AudioConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "siren_classifier.pkl"
SIREN_SAMPLE = ROOT / "data" / "audio" / "siren" / "sound_1.wav"
NON_SIREN_SAMPLE = ROOT / "data" / "audio" / "non-siren" / "sound_401.wav"


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, label: int, positive_probability: float):
        self.label = label
        self.positive_probability = positive_probability

    def predict(self, X):
        return np.asarray([self.label] * len(X))

    def predict_proba(self, X):
        probability = float(self.positive_probability)
        return np.asarray([[1.0 - probability, probability]] * len(X))


class ListSource:
    name = "list"

    def __init__(self, blocks, *, sample_rate: int = 100):
        self.blocks = [np.asarray(block, dtype=np.float32) for block in blocks]
        self.sample_rate = sample_rate
        self._index = 0
        self._open = False
        self._exhausted = False

    @property
    def is_open(self):
        return self._open

    @property
    def exhausted(self):
        return self._exhausted

    def open(self):
        self._open = True
        self._index = 0
        self._exhausted = False

    def read(self):
        if not self._open:
            return None
        if self._index >= len(self.blocks):
            self._exhausted = True
            return None
        block = self.blocks[self._index]
        self._index += 1
        return block

    def close(self):
        self._open = False


def classifier(label: int = 1, probability: float = 0.9) -> SirenClassifier:
    return SirenClassifier(
        FakeModel(label, probability),
        threshold=0.5,
        sample_rate=100,
        n_mfcc=40,
    )


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def test_iter_windows_slides_with_overlap_and_marks_the_tail():
    samples = np.arange(250, dtype=np.float32)

    windows = list(
        iter_windows(samples, sample_rate=100, window_s=1.0, overlap_s=0.5)
    )

    assert [window.index for window in windows] == [0, 1, 2, 3, 4]
    assert [window.start_s for window in windows] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert [window.end_s for window in windows] == [1.0, 1.5, 2.0, 2.5, 2.5]
    assert [window.partial for window in windows] == [False, False, False, False, True]
    assert [window.samples.size for window in windows] == [100, 100, 100, 100, 50]
    assert windows[1].samples[0] == pytest.approx(50.0)
    assert windows[-1].duration_s == pytest.approx(0.5)


def test_iter_windows_can_skip_the_partial_tail():
    samples = np.arange(250, dtype=np.float32)

    windows = list(
        iter_windows(
            samples,
            sample_rate=100,
            window_s=1.0,
            overlap_s=0.5,
            include_partial=False,
        )
    )

    assert len(windows) == 4
    assert all(not window.partial for window in windows)


def test_iter_windows_handles_a_clip_shorter_than_one_window():
    samples = np.arange(50, dtype=np.float32)

    windows = list(
        iter_windows(samples, sample_rate=100, window_s=1.0, overlap_s=0.5)
    )

    assert len(windows) == 1
    assert windows[0].partial is True
    assert windows[0].start_s == 0.0
    assert windows[0].end_s == pytest.approx(0.5)


def test_iter_windows_rejects_invalid_geometry():
    samples = np.zeros(100, dtype=np.float32)

    with pytest.raises(TypeError, match="numpy array"):
        list(iter_windows([0.0], sample_rate=100, window_s=1.0, overlap_s=0.5))
    with pytest.raises(ValueError, match="1-D"):
        list(iter_windows(np.zeros((2, 10)), sample_rate=100, window_s=1.0, overlap_s=0.5))
    with pytest.raises(ValueError, match="empty"):
        list(iter_windows(np.zeros(0), sample_rate=100, window_s=1.0, overlap_s=0.5))
    with pytest.raises(ValueError, match="finite"):
        list(
            iter_windows(
                np.array([0.0, np.nan]), sample_rate=100, window_s=1.0, overlap_s=0.5
            )
        )
    with pytest.raises(ValueError, match="window_s"):
        list(iter_windows(samples, sample_rate=100, window_s=0, overlap_s=0.0))
    with pytest.raises(ValueError, match="overlap_s"):
        list(iter_windows(samples, sample_rate=100, window_s=1.0, overlap_s=1.0))
    with pytest.raises(ValueError, match="sample_rate"):
        list(iter_windows(samples, sample_rate=0, window_s=1.0, overlap_s=0.5))


def test_audio_window_validates_fields():
    assert AudioWindow(0, 0.0, 1.0, np.zeros(10), 100).duration_s == 1.0

    with pytest.raises(ValueError, match="index"):
        AudioWindow(-1, 0.0, 1.0, np.zeros(10), 100)
    with pytest.raises(ValueError, match="start_s"):
        AudioWindow(0, -1.0, 1.0, np.zeros(10), 100)
    with pytest.raises(ValueError, match="end_s"):
        AudioWindow(0, 1.0, 1.0, np.zeros(10), 100)
    with pytest.raises(ValueError, match="numpy array"):
        AudioWindow(0, 0.0, 1.0, [0.0], 100)
    with pytest.raises(ValueError, match="1-D"):
        AudioWindow(0, 0.0, 1.0, np.zeros((2, 2)), 100)
    with pytest.raises(ValueError, match="finite"):
        AudioWindow(0, 0.0, 1.0, np.array([np.nan]), 100)
    with pytest.raises(ValueError, match="sample_rate"):
        AudioWindow(0, 0.0, 1.0, np.zeros(10), 0)


# ---------------------------------------------------------------------------
# File source
# ---------------------------------------------------------------------------
def test_file_source_yields_blocks_then_exhausts(tmp_path):
    path = tmp_path / "tone.wav"
    samples = np.zeros(22050, dtype=np.float32)
    sf.write(path, samples, 22050)

    source = FileAudioSource(path, sample_rate=22050, block_s=0.5)
    source.open()
    blocks = []
    while True:
        block = source.read()
        if block is None:
            break
        blocks.append(block)

    assert len(blocks) == 2
    assert sum(block.size for block in blocks) == 22050
    assert source.exhausted is True

    source.close()
    assert source.is_open is False
    assert source.read() is None


def test_file_source_rejects_missing_file_and_bad_options(tmp_path):
    with pytest.raises(FileNotFoundError, match="audio file not found"):
        FileAudioSource(tmp_path / "missing.wav")

    path = tmp_path / "tone.wav"
    sf.write(path, np.zeros(1000, dtype=np.float32), 22050)

    with pytest.raises(ValueError, match="block_s"):
        FileAudioSource(path, block_s=0)
    with pytest.raises(ValueError, match="block_s"):
        FileAudioSource(path, block_s=float("nan"))
    with pytest.raises(ValueError, match="sample_rate"):
        FileAudioSource(path, sample_rate=0)
    with pytest.raises(ValueError, match="duration"):
        FileAudioSource(path, duration=-1)


# ---------------------------------------------------------------------------
# Streaming predictions
# ---------------------------------------------------------------------------
def test_stream_predictions_matches_iter_windows_geometry():
    samples = np.arange(250, dtype=np.float32)
    source = ListSource([samples[:40], samples[40:90], samples[90:]])

    predictions = list(
        stream_predictions(source, classifier(), window_s=1.0, overlap_s=0.5)
    )
    expected = list(
        iter_windows(samples, sample_rate=100, window_s=1.0, overlap_s=0.5)
    )

    assert [item.window.start_s for item in predictions] == [
        window.start_s for window in expected
    ]
    assert [item.window.end_s for item in predictions] == [
        window.end_s for window in expected
    ]
    assert [item.window.partial for item in predictions] == [
        window.partial for window in expected
    ]
    assert source.is_open is False


def test_stream_predictions_carries_confidence_and_builds_events():
    source = ListSource([np.zeros(200, dtype=np.float32)])

    predictions = list(
        stream_predictions(source, classifier(1, 0.9), window_s=1.0, overlap_s=0.0)
    )

    assert len(predictions) == 2
    assert all(item.is_siren for item in predictions)
    assert predictions[0].confidence == pytest.approx(0.9)

    event = predictions[0].to_event("Road 2")
    assert event.source.value == "audio"
    assert event.detection_type.value == "siren"
    assert event.approach == "Road 2"
    assert event.confidence == pytest.approx(0.9)
    assert event.window_start_s == 0.0
    assert event.window_end_s == pytest.approx(1.0)


def test_event_uses_hard_label_when_confidence_is_missing():
    window = AudioWindow(0, 0.0, 1.0, np.zeros(10), 100)
    prediction = WindowPrediction(window=window, is_siren=True, confidence=None)

    assert prediction.to_event("Road 1").confidence == pytest.approx(1.0)


def test_stream_predictions_rejects_wrong_classifier_type():
    with pytest.raises(TypeError, match="SirenClassifier"):
        list(stream_predictions(ListSource([np.zeros(10)]), object(), window_s=1.0, overlap_s=0.0))


def test_stream_predictions_closes_source_on_error():
    source = ListSource([np.zeros((2, 10), dtype=np.float32)])

    with pytest.raises(ValueError, match="1-D"):
        list(stream_predictions(source, classifier(), window_s=1.0, overlap_s=0.0))

    assert source.is_open is False


def test_summarize_counts_windows_and_confidence():
    window_a = AudioWindow(0, 0.0, 1.0, np.zeros(10), 100)
    window_b = AudioWindow(1, 0.5, 1.5, np.zeros(10), 100)
    predictions = [
        WindowPrediction(window_a, True, 0.8),
        WindowPrediction(window_b, False, 0.2),
    ]

    summary = summarize(predictions)

    assert summary.windows == 2
    assert summary.siren_windows == 1
    assert summary.detected is True
    assert summary.max_confidence == pytest.approx(0.8)
    assert summarize([]).detected is False


# ---------------------------------------------------------------------------
# End-to-end file analysis
# ---------------------------------------------------------------------------
def test_analyze_file_detects_siren_sample():
    if not SIREN_SAMPLE.is_file():
        pytest.skip("siren sample missing")

    model = SirenClassifier.from_path(MODEL_PATH)
    predictions = analyze_file(str(SIREN_SAMPLE), model, AudioConfig())
    summary = summarize(predictions)

    assert summary.windows >= 1
    assert summary.detected is True
    assert summary.max_confidence is not None


def test_analyze_file_does_not_flag_non_siren_sample():
    if not NON_SIREN_SAMPLE.is_file():
        pytest.skip("non-siren sample missing")

    model = SirenClassifier.from_path(MODEL_PATH)
    predictions = analyze_file(str(NON_SIREN_SAMPLE), model, AudioConfig())

    assert len(predictions) >= 1
    assert summarize(predictions).detected is False


def test_analyze_file_requires_matching_sample_rates():
    model = SirenClassifier(FakeModel(1, 0.9), sample_rate=22050)

    with pytest.raises(ValueError, match="sample_rate"):
        analyze_file(str(SIREN_SAMPLE), model, AudioConfig(sample_rate=16000))
