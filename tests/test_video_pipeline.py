from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ambulance_system.video import (
    FramePacket,
    PipelineStats,
    SyntheticVideoSource,
    VideoPipeline,
    VideoSourceError,
)


def frame(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


class FailingOpenSource:
    name = "failing"
    is_open = False
    exhausted = False

    def open(self):
        raise VideoSourceError("camera unavailable")

    def read(self):
        return None

    def close(self):
        pass


class ScriptedSource:
    """Fails the first `failures_first` reads, then yields frames, then exhausts."""

    name = "scripted"

    def __init__(self, frames, failures_first: int = 0):
        self._frames = list(frames)
        self._failures_first = failures_first
        self._open = False
        self._exhausted = False
        self._index = 0
        self._remaining_failures = 0
        self.open_calls = 0
        self.close_calls = 0

    @property
    def is_open(self):
        return self._open

    @property
    def exhausted(self):
        return self._exhausted

    def open(self):
        self.open_calls += 1
        self._open = True
        self._index = 0
        self._exhausted = False
        self._remaining_failures = self._failures_first if self.open_calls == 1 else 0

    def read(self):
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            return None
        if self._index >= len(self._frames):
            self._exhausted = True
            return None
        value = self._frames[self._index]
        self._index += 1
        return value

    def close(self):
        self.close_calls += 1
        self._open = False


class AlwaysFailingSource:
    name = "always-failing"
    is_open = True
    exhausted = False

    def __init__(self):
        self.open_calls = 0
        self.close_calls = 0

    def open(self):
        self.open_calls += 1

    def read(self):
        return None

    def close(self):
        self.close_calls += 1


# ---------------------------------------------------------------------------
# FramePacket / stats
# ---------------------------------------------------------------------------
def test_frame_packet_validates_fields():
    assert FramePacket(0, frame(), True, 0.0).sampled is True

    with pytest.raises(ValueError, match="index"):
        FramePacket(-1, frame(), True, 0.0)
    with pytest.raises(ValueError, match="numpy array"):
        FramePacket(0, [[0]], True, 0.0)
    with pytest.raises(ValueError, match="sampled"):
        FramePacket(0, frame(), 1, 0.0)
    with pytest.raises(ValueError, match="timestamp"):
        FramePacket(0, frame(), True, -0.1)


def test_pipeline_stats_serialises():
    stats = PipelineStats(
        frames_read=10,
        frames_delivered=8,
        frames_dropped=2,
        samples_processed=3,
        errors=1,
        reconnects=0,
        fps=12.5,
        avg_latency_ms=40.0,
        p95_latency_ms=55.0,
        running=False,
        last_error=None,
    )

    payload = stats.to_dict()

    assert payload["frames_read"] == 10
    assert payload["frames_dropped"] == 2
    assert payload["last_error"] is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_constructor_validates_arguments():
    source = SyntheticVideoSource([frame()])

    with pytest.raises(TypeError, match="callable"):
        VideoPipeline(source, None)
    with pytest.raises(ValueError, match="frame_skip"):
        VideoPipeline(source, lambda packet: None, frame_skip=0)
    with pytest.raises(ValueError, match="queue_size"):
        VideoPipeline(source, lambda packet: None, queue_size=0)
    with pytest.raises(ValueError, match="delays"):
        VideoPipeline(source, lambda packet: None, read_delay_s=-0.1)
    with pytest.raises(ValueError, match="delays"):
        VideoPipeline(source, lambda packet: None, reconnect_delay_s=-0.1)
    with pytest.raises(ValueError, match="max_reconnect_attempts"):
        VideoPipeline(source, lambda packet: None, max_reconnect_attempts=-1)


# ---------------------------------------------------------------------------
# Delivery and sampling
# ---------------------------------------------------------------------------
def test_delivers_frames_and_flags_samples():
    frames = [frame(index) for index in range(7)]
    received = []
    pipeline = VideoPipeline(
        SyntheticVideoSource(frames),
        received.append,
        frame_skip=3,
        queue_size=64,
    )

    pipeline.start()
    assert pipeline.wait(timeout=10.0)

    assert [packet.index for packet in received] == list(range(7))
    assert [packet.index for packet in received if packet.sampled] == [0, 3, 6]
    assert pipeline.stats.frames_read == 7
    assert pipeline.stats.frames_delivered == 7
    assert pipeline.stats.samples_processed == 3
    assert pipeline.stats.running is False
    assert pipeline.stats.fps > 0


def test_callback_errors_are_counted_and_do_not_stop_the_pipeline():
    received = []

    def handler(packet):
        received.append(packet.index)
        if packet.index == 1:
            raise RuntimeError("boom")

    pipeline = VideoPipeline(SyntheticVideoSource([frame(i) for i in range(4)]), handler)

    pipeline.start()
    assert pipeline.wait(timeout=10.0)

    assert received == [0, 1, 2, 3]
    assert pipeline.stats.errors == 1


def test_slow_consumer_causes_drops():
    def slow_handler(packet):
        time.sleep(0.12)

    pipeline = VideoPipeline(
        SyntheticVideoSource([frame(i) for i in range(8)]),
        slow_handler,
        queue_size=1,
    )

    pipeline.start()
    assert pipeline.wait(timeout=20.0)

    stats = pipeline.stats
    assert stats.frames_read == 8
    assert stats.frames_dropped > 0
    assert stats.frames_delivered + stats.frames_dropped <= stats.frames_read
    assert stats.errors == 0


# ---------------------------------------------------------------------------
# Reconnection
# ---------------------------------------------------------------------------
def test_reconnects_once_after_read_failure():
    source = ScriptedSource([frame(1), frame(2)], failures_first=1)
    received = []
    pipeline = VideoPipeline(
        source,
        received.append,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=3,
    )

    pipeline.start()
    assert pipeline.wait(timeout=10.0)

    assert [packet.index for packet in received] == [0, 1]
    assert source.open_calls == 2
    assert pipeline.stats.reconnects == 1
    assert pipeline.stats.frames_read == 2


def test_reconnect_limit_stops_the_acquire_loop():
    pipeline = VideoPipeline(
        AlwaysFailingSource(),
        lambda packet: None,
        reconnect_delay_s=0.0,
        max_reconnect_attempts=2,
    )

    pipeline.start()
    assert pipeline.wait(timeout=10.0)

    assert pipeline.stats.reconnects == 2
    assert "reconnect limit" in (pipeline.stats.last_error or "")
    assert pipeline.running is False


def test_start_raises_when_source_cannot_open():
    pipeline = VideoPipeline(FailingOpenSource(), lambda packet: None)

    with pytest.raises(VideoSourceError, match="camera unavailable"):
        pipeline.start()

    assert pipeline.running is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def test_stop_ends_a_running_pipeline():
    pipeline = VideoPipeline(
        SyntheticVideoSource([frame()], repeat=True),
        lambda packet: None,
        read_delay_s=0.0,
    )

    pipeline.start()
    deadline = time.monotonic() + 5.0
    while pipeline.stats.frames_delivered < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert pipeline.running is True

    pipeline.stop(timeout=5.0)

    assert pipeline.running is False
    assert pipeline.wait(timeout=5.0) is True


def test_start_twice_is_rejected_and_restart_works():
    pipeline = VideoPipeline(
        SyntheticVideoSource([frame()], repeat=True),
        lambda packet: None,
        read_delay_s=0.0,
    )
    pipeline.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            pipeline.start()
    finally:
        pipeline.stop(timeout=5.0)

    pipeline.start()
    try:
        assert pipeline.running is True
    finally:
        pipeline.stop(timeout=5.0)


def test_stop_before_start_is_safe():
    pipeline = VideoPipeline(SyntheticVideoSource([frame()]), lambda packet: None)

    pipeline.stop()
    pipeline.stop()

    assert pipeline.running is False


def test_counters_reset_on_restart():
    pipeline = VideoPipeline(SyntheticVideoSource([frame(), frame()]), lambda packet: None)

    pipeline.start()
    assert pipeline.wait(timeout=10.0)
    assert pipeline.stats.frames_read == 2

    pipeline.start()
    assert pipeline.wait(timeout=10.0)
    assert pipeline.stats.frames_read == 2
    assert pipeline.stats.frames_delivered == 2
