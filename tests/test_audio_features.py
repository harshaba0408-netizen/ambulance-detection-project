from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from ambulance_system.audio import load_audio, mfcc_features

ROOT = Path(__file__).resolve().parents[1]
SIREN_SAMPLE = ROOT / "data" / "audio" / "siren" / "sound_1.wav"


def tone(seconds: float = 1.0, sample_rate: int = 48000, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    mono = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return np.column_stack([mono] * channels)


def test_load_audio_resamples_and_downmixes(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, tone(1.0, 48000, 2), 48000)

    samples, rate = load_audio(path, sample_rate=22050)

    assert rate == 22050
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert samples.size == pytest.approx(22050, abs=2)


def test_load_audio_rejects_invalid_options(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, tone(), 48000)

    with pytest.raises(ValueError, match="sample_rate"):
        load_audio(path, sample_rate=0)
    with pytest.raises(ValueError, match="duration"):
        load_audio(path, duration=0)
    with pytest.raises(ValueError, match="path"):
        load_audio("   ")


def test_mfcc_matches_the_legacy_feature_pipeline():
    if not SIREN_SAMPLE.is_file():
        pytest.skip("siren sample missing")

    samples, rate = load_audio(SIREN_SAMPLE, duration=3.0)
    features = mfcc_features(samples, sample_rate=rate, n_mfcc=40)

    y, sr = librosa.load(str(SIREN_SAMPLE), duration=3.0)
    legacy = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    assert features.shape == (40,)
    assert np.all(np.isfinite(features))
    np.testing.assert_allclose(features, legacy, rtol=1e-5, atol=1e-5)


def test_mfcc_is_deterministic():
    samples = np.random.default_rng(0).normal(size=22050).astype(np.float32)

    first = mfcc_features(samples, sample_rate=22050)
    second = mfcc_features(samples, sample_rate=22050)

    np.testing.assert_array_equal(first, second)


def test_mfcc_supports_a_different_dimension():
    samples = np.zeros(22050, dtype=np.float32)

    assert mfcc_features(samples, n_mfcc=13).shape == (13,)


def test_mfcc_rejects_invalid_input():
    samples = np.zeros(22050, dtype=np.float32)

    with pytest.raises(TypeError, match="numpy array"):
        mfcc_features([0.0] * 100)
    with pytest.raises(ValueError, match="1-D"):
        mfcc_features(np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        mfcc_features(np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        mfcc_features(np.array([0.0, np.nan, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="sample_rate"):
        mfcc_features(samples, sample_rate=0)
    with pytest.raises(ValueError, match="n_mfcc"):
        mfcc_features(samples, n_mfcc=0)
