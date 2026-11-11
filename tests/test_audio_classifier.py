from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ambulance_system.audio import SirenClassifier, SirenPrediction, load_audio

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "siren_classifier.pkl"
SIREN_SAMPLE = ROOT / "data" / "audio" / "siren" / "sound_1.wav"
NON_SIREN_SAMPLE = ROOT / "data" / "audio" / "non-siren" / "sound_401.wav"


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, label: int, positive_probability: float = 0.5, *, proba_error: bool = False):
        self.label = label
        self.positive_probability = positive_probability
        self.proba_error = proba_error

    def predict(self, X):
        return np.asarray([self.label] * len(X))

    def predict_proba(self, X):
        if self.proba_error:
            raise RuntimeError("no probabilities")
        probability = float(self.positive_probability)
        return np.asarray([[1.0 - probability, probability]] * len(X))


class NoProbaModel:
    classes_ = np.array([0, 1])

    def __init__(self, label: int):
        self.label = label

    def predict(self, X):
        return np.asarray([self.label] * len(X))


def features(value: float = 0.0) -> np.ndarray:
    return np.full(40, value, dtype=float)


# ---------------------------------------------------------------------------
# Real checkpoint
# ---------------------------------------------------------------------------
def test_published_classifier_loads_and_reports_classes():
    classifier = SirenClassifier.from_path(MODEL_PATH)

    assert classifier.classes == [0, 1]
    assert classifier.threshold == pytest.approx(0.5)
    assert classifier.sample_rate == 22050


def test_published_classifier_separates_siren_and_non_siren_samples():
    if not SIREN_SAMPLE.is_file() or not NON_SIREN_SAMPLE.is_file():
        pytest.skip("audio samples missing")

    classifier = SirenClassifier.from_path(MODEL_PATH)
    siren, rate = load_audio(SIREN_SAMPLE)
    non_siren, _ = load_audio(NON_SIREN_SAMPLE)

    siren_prediction = classifier.predict_samples(siren)
    non_siren_prediction = classifier.predict_samples(non_siren)

    assert siren_prediction.is_siren is True
    assert siren_prediction.confidence is not None
    assert siren_prediction.confidence >= 0.5
    assert non_siren_prediction.is_siren is False
    assert non_siren_prediction.confidence is not None
    assert non_siren_prediction.confidence < 0.5


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="siren classifier missing"):
        SirenClassifier.from_path(tmp_path / "missing.pkl")


# ---------------------------------------------------------------------------
# Threshold policy and probability handling
# ---------------------------------------------------------------------------
def test_threshold_can_override_the_hard_label():
    classifier = SirenClassifier(FakeModel(label=0, positive_probability=0.6), threshold=0.5)
    assert classifier.predict_features(features()).is_siren is True

    strict = SirenClassifier(FakeModel(label=1, positive_probability=0.6), threshold=0.7)
    prediction = strict.predict_features(features())

    assert prediction.is_siren is False
    assert prediction.confidence == pytest.approx(0.6)


def test_model_without_predict_proba_falls_back_to_the_label():
    classifier = SirenClassifier(NoProbaModel(label=1))

    prediction = classifier.predict_features(features())

    assert prediction.is_siren is True
    assert prediction.confidence is None
    assert prediction.label == 1


def test_predict_proba_failure_falls_back_to_the_label():
    classifier = SirenClassifier(FakeModel(label=0, proba_error=True))

    prediction = classifier.predict_features(features())

    assert prediction.is_siren is False
    assert prediction.confidence is None


def test_invalid_constructor_values():
    with pytest.raises(TypeError, match="predict"):
        SirenClassifier(None)
    with pytest.raises(TypeError, match="predict"):
        SirenClassifier(object())
    with pytest.raises(ValueError, match="threshold"):
        SirenClassifier(FakeModel(0), threshold=1.5)
    with pytest.raises(ValueError, match="sample_rate"):
        SirenClassifier(FakeModel(0), sample_rate=0)
    with pytest.raises(ValueError, match="n_mfcc"):
        SirenClassifier(FakeModel(0), n_mfcc=0)


def test_feature_input_is_validated():
    classifier = SirenClassifier(FakeModel(0))

    with pytest.raises(ValueError, match="1-D"):
        classifier.predict_features(np.zeros((2, 40)))
    with pytest.raises(ValueError, match="empty"):
        classifier.predict_features(np.zeros(0))
    with pytest.raises(ValueError, match="finite"):
        classifier.predict_features(np.full(40, np.nan))


def test_prediction_validates_confidence():
    assert SirenPrediction(True, 0.9, 1).is_siren is True
    assert SirenPrediction(False, None, 0).confidence is None

    with pytest.raises(ValueError, match="confidence"):
        SirenPrediction(True, 1.2, 1)
    with pytest.raises(ValueError, match="label"):
        SirenPrediction(True, 0.9, 1.5)
    with pytest.raises(ValueError, match="is_siren"):
        SirenPrediction(1, 0.9, 1)
