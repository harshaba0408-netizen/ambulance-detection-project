"""Audio loading and MFCC feature extraction.

There is exactly one feature implementation so the dashboard and the streaming
pipeline cannot drift apart. It matches the original one-shot code:

```python
y, sr = librosa.load(path, duration=3.0)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
feat = np.mean(mfccs.T, axis=0)
```

No model is loaded here and no classification happens here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

DEFAULT_N_MFCC = 40


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = 22050,
    duration: Optional[float] = None,
) -> tuple[np.ndarray, int]:
    """Load a file as mono float32 at `sample_rate`.

    This mirrors the old `librosa.load(path, duration=3.0)` call: librosa's
    default sample rate is 22050 Hz and `mono=True` is the default.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive int")
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise ValueError("duration must be a positive finite number or None")

    source = str(path)
    if not source.strip():
        raise ValueError("audio path must be a non-empty string")

    samples, rate = librosa.load(source, sr=sample_rate, mono=True, duration=duration)
    return np.asarray(samples, dtype=np.float32), int(rate)


def mfcc_features(
    samples: np.ndarray,
    *,
    sample_rate: int = 22050,
    n_mfcc: int = DEFAULT_N_MFCC,
) -> np.ndarray:
    """Return the 40-dimensional mean-MFCC vector for one window."""
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a numpy array")
    if samples.ndim != 1:
        raise ValueError("samples must be a 1-D mono array")
    if samples.size == 0:
        raise ValueError("samples must not be empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive int")
    if not isinstance(n_mfcc, int) or isinstance(n_mfcc, bool) or n_mfcc <= 0:
        raise ValueError("n_mfcc must be a positive int")

    mfccs = librosa.feature.mfcc(
        y=np.asarray(samples, dtype=np.float32),
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )
    return np.mean(mfccs.T, axis=0)
