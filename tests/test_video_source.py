from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ambulance_system.video import (
    CaptureVideoSource,
    FileVideoSource,
    RTSPVideoSource,
    SyntheticVideoSource,
    VideoSourceError,
    WebcamVideoSource,
    create_source,
    redact_url,
)
from ambulance_system.config import VideoConfig


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in."""

    def __init__(self, frames, *, opened=True):
        self._frames = list(frames)
        self._index = 0
        self._opened = opened
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self._opened and not self.released

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def read(self):
        if self.released or self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self):
        self.released = True


def frame(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not a real video")
    return path


# ---------------------------------------------------------------------------
# File source
# ---------------------------------------------------------------------------
def test_missing_file_is_rejected_before_opening(tmp_path):
    with pytest.raises(FileNotFoundError, match="video file not found"):
        FileVideoSource(tmp_path / "missing.mp4")


def test_file_source_reads_frames_until_exhausted(tmp_path):
    capture = FakeCapture([frame(1), frame(2)])
    source = FileVideoSource(video_file(tmp_path), capture_factory=lambda: capture)

    source.open()
    assert source.is_open
    assert source.read() is not None
    assert source.read() is not None
    assert source.read() is None
    assert source.exhausted is True

    source.close()
    assert not source.is_open
    assert capture.released is True


def test_file_source_requests_single_frame_buffer(tmp_path):
    capture = FakeCapture([])
    source = FileVideoSource(video_file(tmp_path), capture_factory=lambda: capture)

    source.open()

    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in capture.set_calls


def test_loop_rewinds_after_end_of_stream(tmp_path):
    captures = []

    def factory():
        capture = FakeCapture([frame(1)])
        captures.append(capture)
        return capture

    source = FileVideoSource(video_file(tmp_path), loop=True, capture_factory=factory)

    source.open()

    assert source.read() is not None
    assert source.read() is not None  # rewound by reopening the file
    assert len(captures) == 2
    assert source.exhausted is False


def test_open_failure_raises_video_source_error(tmp_path):
    source = FileVideoSource(
        video_file(tmp_path), capture_factory=lambda: FakeCapture([], opened=False)
    )

    with pytest.raises(VideoSourceError, match="could not open"):
        source.open()


def test_close_is_idempotent(tmp_path):
    source = FileVideoSource(
        video_file(tmp_path), capture_factory=lambda: FakeCapture([frame()])
    )
    source.open()

    source.close()
    source.close()

    assert not source.is_open


def test_read_before_open_returns_none(tmp_path):
    source = FileVideoSource(
        video_file(tmp_path), capture_factory=lambda: FakeCapture([frame()])
    )

    assert source.read() is None


# ---------------------------------------------------------------------------
# Webcam / RTSP / factory
# ---------------------------------------------------------------------------
def test_webcam_rejects_negative_index():
    with pytest.raises(ValueError, match="webcam index"):
        WebcamVideoSource(-1)


def test_webcam_opens_injected_capture():
    capture = FakeCapture([frame()])
    source = WebcamVideoSource(1, capture_factory=lambda: capture)

    source.open()

    assert source.name == "webcam:1"
    assert source.read() is not None


def test_rtsp_rejects_empty_url_and_bad_timeout():
    with pytest.raises(ValueError, match="RTSP URL"):
        RTSPVideoSource("   ")
    with pytest.raises(ValueError, match="read_timeout_s"):
        RTSPVideoSource("rtsp://example/stream", read_timeout_s=0)


def test_rtsp_url_password_is_redacted():
    source = RTSPVideoSource(
        "rtsp://user:secret@camera.local:554/stream",
        capture_factory=lambda: FakeCapture([frame()]),
    )

    assert "secret" not in source.name
    assert "user:***@" in source.name


def test_redact_url_keeps_urls_without_credentials():
    assert redact_url("rtsp://camera.local/stream") == "rtsp://camera.local/stream"


def test_create_source_builds_file_source(tmp_path):
    path = video_file(tmp_path)
    config = VideoConfig(source_type="file", file_path=str(path))

    source = create_source(config, capture_factory=lambda: FakeCapture([frame()]))

    assert isinstance(source, FileVideoSource)
    assert source.path == path


def test_create_source_requires_file_path():
    config = VideoConfig(source_type="file", file_path=None)

    with pytest.raises(ValueError, match="file_path"):
        create_source(config, capture_factory=lambda: FakeCapture([frame()]))


def test_create_source_builds_webcam_and_rtsp():
    webcam = create_source(
        VideoConfig(source_type="webcam", webcam_index=2),
        capture_factory=lambda: FakeCapture([frame()]),
    )
    rtsp = create_source(
        VideoConfig(source_type="rtsp", rtsp_url="rtsp://camera.local/stream"),
        capture_factory=lambda: FakeCapture([frame()]),
    )

    assert isinstance(webcam, WebcamVideoSource)
    assert webcam.index == 2
    assert isinstance(rtsp, RTSPVideoSource)


def test_capture_video_source_is_open_after_open():
    capture = FakeCapture([frame()])
    source = CaptureVideoSource(lambda: capture, name="test")

    source.open()

    assert source.is_open
    assert source.name == "test"


# ---------------------------------------------------------------------------
# Synthetic source (used by pipeline tests)
# ---------------------------------------------------------------------------
def test_synthetic_source_iterates_then_exhausts():
    source = SyntheticVideoSource([frame(1), frame(2)])

    source.open()

    assert source.read() is not None
    assert source.read() is not None
    assert source.read() is None
    assert source.exhausted is True


def test_synthetic_source_can_repeat():
    source = SyntheticVideoSource([frame(1)], repeat=True)

    source.open()

    assert source.read() is not None
    assert source.read() is not None
    assert source.exhausted is False


def test_synthetic_source_requires_frames():
    with pytest.raises(ValueError, match="at least one frame"):
        SyntheticVideoSource([])
    with pytest.raises(ValueError, match="numpy arrays"):
        SyntheticVideoSource([[1, 2, 3]])


def test_synthetic_source_closed_returns_none():
    source = SyntheticVideoSource([frame()])

    assert source.read() is None
