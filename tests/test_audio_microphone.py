from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ambulance_system.audio import (
    AudioDeviceInfo,
    AudioSourceError,
    MicrophoneAudioSource,
    SirenClassifier,
    list_input_devices,
    stream_predictions,
)


# ---------------------------------------------------------------------------
# Test doubles: no sounddevice backend and no hardware required
# ---------------------------------------------------------------------------
class FakeStream:
    def __init__(self, callback):
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def emit(self, block, status=None):
        data = np.asarray(block, dtype=np.float32)
        self.callback(data, data.shape[0], None, status)


class FakeFactory:
    def __init__(self):
        self.calls = []
        self.stream = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.stream = FakeStream(kwargs["callback"])
        return self.stream


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, label=1, positive_probability=0.9):
        self.label = label
        self.positive_probability = positive_probability

    def predict(self, X):
        return np.asarray([self.label] * len(X))

    def predict_proba(self, X):
        probability = float(self.positive_probability)
        return np.asarray([[1.0 - probability, probability]] * len(X))


def classifier(label: int = 1, probability: float = 0.9) -> SirenClassifier:
    return SirenClassifier(
        FakeModel(label, probability),
        threshold=0.5,
        sample_rate=100,
        n_mfcc=40,
    )


def make_source(factory, **kwargs) -> MicrophoneAudioSource:
    options = {
        "sample_rate": 100,
        "block_s": 0.25,
        "buffer_blocks": 8,
        "read_timeout_s": 0.5,
        "stream_factory": factory,
    }
    options.update(kwargs)
    return MicrophoneAudioSource(**options)


# ---------------------------------------------------------------------------
# Construction and naming
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": True},
        {"sample_rate": 22050.0},
        {"block_s": 0},
        {"block_s": float("nan")},
        {"channels": 3},
        {"channels": True},
        {"buffer_blocks": 0},
        {"buffer_blocks": 1.5},
        {"read_timeout_s": 0},
        {"read_timeout_s": float("inf")},
        {"device": 1.5},
        {"stream_factory": 42},
    ],
)
def test_invalid_construction_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MicrophoneAudioSource(**kwargs)


def test_name_reflects_device_selection():
    factory = FakeFactory()
    assert make_source(factory).name == "mic:default"
    assert make_source(factory, device=1).name == "mic:1"
    assert make_source(factory, device="Realtek").name == "mic:Realtek"


def test_default_geometry_is_derived_from_seconds():
    source = make_source(FakeFactory(), sample_rate=22050, block_s=0.25, channels=2)
    assert source.sample_rate == 22050
    assert source.block_samples == 5512
    assert source.channels == 2
    assert not source.is_open
    assert not source.exhausted


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def test_open_passes_expected_arguments_and_starts_stream():
    factory = FakeFactory()
    source = make_source(factory, device=None, channels=1)

    source.open()

    assert source.is_open
    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call["device"] is None
    assert call["sample_rate"] == 100
    assert call["block_samples"] == 25
    assert call["channels"] == 1
    assert callable(call["callback"])
    assert factory.stream.started


def test_open_is_idempotent():
    factory = FakeFactory()
    source = make_source(factory)
    source.open()
    source.open()
    assert len(factory.calls) == 1


def test_read_before_open_returns_none():
    source = make_source(FakeFactory())
    assert source.read() is None


def test_callback_blocks_are_returned_and_close_releases_resources():
    factory = FakeFactory()
    source = make_source(factory)
    source.open()

    factory.stream.emit(np.arange(25, dtype=np.float32) / 25.0)
    block = source.read()

    assert block is not None
    assert block.dtype == np.float32
    assert block.shape == (25,)
    assert block[0] == np.float32(0.0)
    assert block[-1] == pytest.approx(0.96)

    source.close()

    assert not source.is_open
    assert source.exhausted
    assert factory.stream.stopped
    assert factory.stream.closed
    assert source.read() is None


def test_stereo_input_is_downmixed_to_mono():
    factory = FakeFactory()
    source = make_source(factory, channels=2)
    source.open()

    factory.stream.emit(np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32))
    block = source.read()

    assert block is not None
    assert block.shape == (2,)
    np.testing.assert_allclose(block, np.array([2.0, 3.0], dtype=np.float32))


def test_callback_status_is_recorded():
    factory = FakeFactory()
    source = make_source(factory)
    source.open()

    factory.stream.emit(np.zeros(25, dtype=np.float32), status="input overflow")
    source.read()

    assert source.last_status == "input overflow"


def test_invalid_blocks_are_rejected_and_counted():
    factory = FakeFactory()
    source = make_source(factory, read_timeout_s=0.05)
    source.open()

    factory.stream.emit(np.array([], dtype=np.float32))
    factory.stream.emit(np.array([np.nan, 1.0], dtype=np.float32))
    factory.stream.emit(np.array([1e24, -1e24], dtype=np.float32))

    assert source.invalid_blocks == 3
    with pytest.raises(AudioSourceError, match="rejected 3 invalid block"):
        source.read()


def test_plausible_audio_above_unit_range_is_accepted():
    factory = FakeFactory()
    source = make_source(factory)
    source.open()

    factory.stream.emit(np.array([1.5, -2.0], dtype=np.float32))
    block = source.read()

    assert block is not None
    np.testing.assert_allclose(block, np.array([1.5, -2.0], dtype=np.float32))
    assert source.invalid_blocks == 0


def test_read_raises_when_the_backend_stalls():
    factory = FakeFactory()
    source = make_source(factory, read_timeout_s=0.01)
    source.open()

    with pytest.raises(AudioSourceError, match="no audio received"):
        source.read()


def test_close_unblocks_a_waiting_reader():
    factory = FakeFactory()
    source = make_source(factory, read_timeout_s=5.0)
    source.open()

    result = {}

    def reader():
        result["block"] = source.read()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    time.sleep(0.05)
    source.close()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result["block"] is None


def test_a_full_buffer_drops_the_newest_block_and_counts_it():
    factory = FakeFactory()
    source = make_source(factory, buffer_blocks=1)
    source.open()

    factory.stream.emit(np.array([1.0, 1.0], dtype=np.float32))
    factory.stream.emit(np.array([2.0, 2.0], dtype=np.float32))

    assert source.dropped_blocks == 1
    block = source.read()
    assert block is not None
    np.testing.assert_allclose(block, np.array([1.0, 1.0], dtype=np.float32))


def test_open_failure_is_wrapped_and_stream_is_closed():
    class BrokenFactory:
        def __init__(self):
            self.stream = FakeStream(lambda *args: None)

        def __call__(self, **kwargs):
            return self.stream

        def start(self):
            raise OSError("device busy")

    factory = BrokenFactory()
    factory.stream.start = factory.start
    source = make_source(factory)

    with pytest.raises(AudioSourceError, match="could not open microphone source"):
        source.open()

    assert not source.is_open
    assert factory.stream.closed


# ---------------------------------------------------------------------------
# Device listing
# ---------------------------------------------------------------------------
def test_list_input_devices_hides_output_only_entries():
    raw = [
        {"name": "Mic A", "max_input_channels": 2, "default_samplerate": 48000.0},
        {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
        {"name": "Mic B", "max_input_channels": 1, "default_samplerate": 44100.0},
    ]

    devices = list_input_devices(lambda: raw)

    assert [device.index for device in devices] == [0, 2]
    assert [device.name for device in devices] == ["Mic A", "Mic B"]
    assert devices[0].max_input_channels == 2
    assert devices[1].default_samplerate == pytest.approx(44100.0)
    assert all(isinstance(device, AudioDeviceInfo) for device in devices)


def test_list_input_devices_wraps_backend_errors():
    def broken():
        raise RuntimeError("PortAudio unavailable")

    with pytest.raises(AudioSourceError, match="could not list audio devices"):
        list_input_devices(broken)


def test_audio_device_info_validates_fields():
    with pytest.raises(ValueError):
        AudioDeviceInfo(index=-1, name="x", max_input_channels=1, default_samplerate=48000.0)
    with pytest.raises(ValueError):
        AudioDeviceInfo(index=0, name="x", max_input_channels=0, default_samplerate=48000.0)
    with pytest.raises(ValueError):
        AudioDeviceInfo(index=0, name="x", max_input_channels=1, default_samplerate=0.0)


# ---------------------------------------------------------------------------
# Integration with the shared windowing pipeline
# ---------------------------------------------------------------------------
def test_microphone_blocks_feed_the_shared_stream_pipeline():
    factory = FakeFactory()
    source = make_source(factory, sample_rate=100, block_s=0.25)
    source.open()

    generator = stream_predictions(
        source,
        classifier(probability=0.9),
        window_s=1.0,
        overlap_s=0.5,
    )
    try:
        for _ in range(4):
            factory.stream.emit(np.full(25, 0.25, dtype=np.float32))

        prediction = next(generator)

        assert prediction.window.index == 0
        assert prediction.window.start_s == pytest.approx(0.0)
        assert prediction.window.end_s == pytest.approx(1.0)
        assert prediction.window.samples.size == 100
        assert prediction.is_siren
        assert prediction.confidence == pytest.approx(0.9)
        assert not prediction.window.partial
    finally:
        generator.close()
        source.close()

    assert not source.is_open
