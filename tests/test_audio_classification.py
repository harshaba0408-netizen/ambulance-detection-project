import os
import pickle
import pytest
import numpy as np
import librosa
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "siren_classifier.pkl")
SIREN_SAMPLE = os.path.join(BASE_DIR, "data", "audio", "siren", "sound_1.wav")
NON_SIREN_SAMPLE = os.path.join(BASE_DIR, "data", "audio", "non-siren", "sound_401.wav")


def extract_mfcc(audio_path, duration=3):
    y, sr = librosa.load(audio_path, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


def test_siren_classifier_exists():
    assert os.path.exists(MODEL_PATH), f"Classifier file missing: {MODEL_PATH}"


def test_siren_classifier_loading():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
    assert clf is not None
    assert hasattr(clf, "predict")
    assert hasattr(clf, "n_classes_")
    assert clf.n_classes_ == 2


def test_mfcc_feature_extraction_dimension():
    if not os.path.exists(SIREN_SAMPLE):
        pytest.skip("Siren audio sample missing")

    features = extract_mfcc(SIREN_SAMPLE, duration=3)
    assert isinstance(features, np.ndarray)
    assert features.shape == (40,), f"Expected shape (40,), got {features.shape}"
    assert not np.isnan(features).any(), "MFCC features contain NaN values"


def test_siren_classification_prediction():
    if not os.path.exists(SIREN_SAMPLE) or not os.path.exists(NON_SIREN_SAMPLE):
        pytest.skip("Audio samples missing")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)

    siren_feat = extract_mfcc(SIREN_SAMPLE)
    siren_pred = clf.predict([siren_feat])[0]
    assert siren_pred == 1, f"Expected Siren class (1), got {siren_pred}"

    non_siren_feat = extract_mfcc(NON_SIREN_SAMPLE)
    non_siren_pred = clf.predict([non_siren_feat])[0]
    assert non_siren_pred == 0, f"Expected Non-Siren class (0), got {non_siren_pred}"


def test_invalid_audio_path_handling():
    with pytest.raises(Exception):
        _ = extract_mfcc("non_existent_audio_path_12345.wav")
