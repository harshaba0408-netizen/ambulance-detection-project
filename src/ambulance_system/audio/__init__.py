"""Audio package: features, classifier wrapper and offline/live streaming."""

from ambulance_system.audio.classifier import SirenClassifier, SirenPrediction
from ambulance_system.audio.devices import (
    AudioDeviceInfo,
    AudioSource,
    AudioSourceError,
    FileAudioSource,
    MicrophoneAudioSource,
    list_input_devices,
)
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
    "AudioDeviceInfo",
    "AudioSource",
    "AudioSourceError",
    "AudioWindow",
    "DEFAULT_N_MFCC",
    "FileAudioSource",
    "MicrophoneAudioSource",
    "SirenClassifier",
    "SirenPrediction",
    "StreamSummary",
    "WindowPrediction",
    "analyze_file",
    "iter_windows",
    "list_input_devices",
    "load_audio",
    "mfcc_features",
    "stream_predictions",
    "summarize",
]
