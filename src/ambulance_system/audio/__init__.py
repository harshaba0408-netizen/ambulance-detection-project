"""Audio package: features, classifier wrapper and offline streaming."""

from ambulance_system.audio.classifier import SirenClassifier, SirenPrediction
from ambulance_system.audio.devices import AudioSource, AudioSourceError, FileAudioSource
from ambulance_system.audio.features import DEFAULT_N_MFCC, load_audio, mfcc_features
from ambulance_system.audio.streaming import (
    AudioWindow,
    StreamSummary,
    WindowPrediction,
    analyze_file,
    iter_windows,
    stream_predictions,
    summarize,
)

__all__ = [
    "AudioSource",
    "AudioSourceError",
    "AudioWindow",
    "DEFAULT_N_MFCC",
    "FileAudioSource",
    "SirenClassifier",
    "SirenPrediction",
    "StreamSummary",
    "WindowPrediction",
    "analyze_file",
    "iter_windows",
    "load_audio",
    "mfcc_features",
    "stream_predictions",
    "summarize",
]
