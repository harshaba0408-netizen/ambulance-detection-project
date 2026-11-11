"""Wrapper around the pickled siren classifier.

The original dashboard called `audio_clf.predict([feat])[0]` directly and only
ever saw a hard label. This wrapper keeps that behaviour, validates inputs, and
additionally exposes the positive-class probability when the estimator supports
`predict_proba` (the published RandomForest does).

The positive label is `1`, matching the existing data layout and tests.
"""

from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ambulance_system.audio.features import DEFAULT_N_MFCC, mfcc_features

SIREN_LABEL = 1


@dataclass(frozen=True)
class SirenPrediction:
    """One classification result for one audio window."""

    is_siren: bool
    confidence: Optional[float]
    label: int

    def __post_init__(self) -> None:
        if not isinstance(self.is_siren, bool):
            raise ValueError("is_siren must be a bool")
        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not math.isfinite(float(self.confidence))
                or not 0.0 <= float(self.confidence) <= 1.0
            ):
                raise ValueError("confidence must be within [0.0, 1.0] or None")
        if isinstance(self.label, bool) or not isinstance(self.label, int):
            raise ValueError("label must be an int")


class SirenClassifier:
    """Validated inference wrapper for a binary siren estimator."""

    def __init__(
        self,
        model: Any,
        *,
        threshold: float = 0.5,
        sample_rate: int = 22050,
        n_mfcc: int = DEFAULT_N_MFCC,
    ) -> None:
        if model is None or not hasattr(model, "predict"):
            raise TypeError("model must expose predict()")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("threshold must be a finite number within [0.0, 1.0]")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive int")
        if not isinstance(n_mfcc, int) or isinstance(n_mfcc, bool) or n_mfcc <= 0:
            raise ValueError("n_mfcc must be a positive int")

        self._model = model
        self._threshold = float(threshold)
        self._sample_rate = sample_rate
        self._n_mfcc = n_mfcc
        self._classes = [int(item) for item in np.asarray(getattr(model, "classes_", [])).reshape(-1)]

    @classmethod
    def from_path(cls, path: str | Path, **kwargs: Any) -> "SirenClassifier":
        """Load a trusted pickle. Do not use this on an untrusted file."""
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"siren classifier missing: {model_path}")
        with warnings.catch_warnings():
            # sklearn warns when a pickle was written by a different version.
            warnings.simplefilter("ignore")
            with model_path.open("rb") as handle:
                model = pickle.load(handle)
        return cls(model, **kwargs)

    # ------------------------------------------------------------------
    @property
    def model(self) -> Any:
        return self._model

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def n_mfcc(self) -> int:
        return self._n_mfcc

    @property
    def classes(self) -> list[int]:
        return list(self._classes)

    # ------------------------------------------------------------------
    def predict_features(self, features: np.ndarray) -> SirenPrediction:
        """Classify one pre-computed feature vector."""
        vector = np.asarray(features, dtype=float)
        if vector.ndim != 1:
            raise ValueError("features must be a 1-D array")
        if vector.size == 0:
            raise ValueError("features must not be empty")
        if not np.all(np.isfinite(vector)):
            raise ValueError("features must contain only finite values")

        label = int(np.asarray(self._model.predict([vector])).reshape(-1)[0])
        confidence = self._positive_probability(vector)
        if confidence is not None:
            is_siren = confidence >= self._threshold
        else:
            is_siren = label == SIREN_LABEL
        return SirenPrediction(is_siren=is_siren, confidence=confidence, label=label)

    def predict_samples(self, samples: np.ndarray) -> SirenPrediction:
        """Extract MFCCs from one window and classify it."""
        features = mfcc_features(
            samples,
            sample_rate=self._sample_rate,
            n_mfcc=self._n_mfcc,
        )
        return self.predict_features(features)

    # ------------------------------------------------------------------
    def _positive_probability(self, vector: np.ndarray) -> Optional[float]:
        if not hasattr(self._model, "predict_proba"):
            return None
        try:
            row = np.asarray(self._model.predict_proba([vector]))
        except Exception:
            return None
        if row.ndim != 2 or row.shape[0] != 1:
            return None
        row = row[0]
        if self._classes and SIREN_LABEL in self._classes:
            index = self._classes.index(SIREN_LABEL)
        elif row.size == 2:
            index = 1
        else:
            return None
        if index >= row.size:
            return None
        value = float(row[index])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value
