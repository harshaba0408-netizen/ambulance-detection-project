"""Video source abstractions: file, webcam, RTSP and synthetic.

All sources share a tiny interface so the pipeline and tests never care where
frames come from. OpenCV capture creation is injectable, which keeps tests
deterministic and offline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from ambulance_system.config import VideoConfig


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened."""


@runtime_checkable
class VideoSource(Protocol):
    """Minimal frame source interface."""

    @property
    def name(self) -> str:
        """Human-readable source name, safe to log."""

    @property
    def is_open(self) -> bool:
        """True while the underlying capture is open."""

    @property
    def exhausted(self) -> bool:
        """True when a finite source has no more frames."""

    def open(self) -> None:
        """Open the source. Raises VideoSourceError on failure."""

    def read(self) -> Optional[np.ndarray]:
        """Return the next frame, or None on end-of-stream/failure."""

    def close(self) -> None:
        """Release the underlying resource. Idempotent."""


def redact_url(url: str) -> str:
    """Hide a password embedded in a URL before it reaches a log line."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid url>"
    if parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return url


class CaptureVideoSource:
    """Base class for OpenCV-backed sources."""

    def __init__(self, opener: Callable[[], Any], *, name: str) -> None:
        self._opener = opener
        self._name = name
        self._capture: Any = None
        self._exhausted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def open(self) -> None:
        if self._capture is not None:
            return
        capture = self._opener()
        if capture is None or not capture.isOpened():
            try:
                if capture is not None:
                    capture.release()
            except Exception:
                pass
            raise VideoSourceError(f"could not open video source: {self._name}")
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._capture = capture
        self._exhausted = False

    def read(self) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


class FileVideoSource(CaptureVideoSource):
    """Recorded video file. Optionally loops back to the first frame."""

    def __init__(
        self,
        path: str | Path,
        *,
        loop: bool = False,
        capture_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        self.loop = loop
        factory = capture_factory or (lambda: cv2.VideoCapture(str(self.path)))
        super().__init__(factory, name=f"file:{self.path.name}")

    def read(self) -> Optional[np.ndarray]:
        frame = super().read()
        if frame is not None:
            return frame

        if self.loop and self.is_open:
            self.close()
            self.open()
            frame = super().read()
            if frame is not None:
                return frame

        self._exhausted = True
        return None


class WebcamVideoSource(CaptureVideoSource):
    """Local camera index."""

    def __init__(
        self,
        index: int = 0,
        *,
        capture_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("webcam index must be a non-negative int")
        self.index = index
        factory = capture_factory or (lambda: cv2.VideoCapture(index))
        super().__init__(factory, name=f"webcam:{index}")


class RTSPVideoSource(CaptureVideoSource):
    """Network stream.

    RTSP has only been exercised with an injected capture in tests. It is not
    verified against a real camera because no authorised test stream exists.
    """

    def __init__(
        self,
        url: str,
        *,
        read_timeout_s: float = 5.0,
        capture_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("RTSP URL must be a non-empty string")
        if read_timeout_s <= 0:
            raise ValueError("read_timeout_s must be > 0")
        self.url = url
        self.read_timeout_s = float(read_timeout_s)
        factory = capture_factory or (lambda: cv2.VideoCapture(url))
        super().__init__(factory, name=f"rtsp:{redact_url(url)}")


class SyntheticVideoSource:
    """In-memory frame source used by tests and offline demos."""

    def __init__(self, frames: Iterable[np.ndarray], *, repeat: bool = False) -> None:
        self._frames = list(frames)
        if not self._frames:
            raise ValueError("at least one frame is required")
        if any(not isinstance(frame, np.ndarray) for frame in self._frames):
            raise ValueError("all frames must be numpy arrays")
        self.repeat = repeat
        self._index = 0
        self._open = False
        self._exhausted = False

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def open(self) -> None:
        self._index = 0
        self._open = True
        self._exhausted = False

    def read(self) -> Optional[np.ndarray]:
        if not self._open:
            return None
        if self._index >= len(self._frames):
            if not self.repeat:
                self._exhausted = True
                return None
            self._index = 0
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def close(self) -> None:
        self._open = False


def create_source(config: VideoConfig, *, capture_factory: Optional[Callable[[], Any]] = None) -> VideoSource:
    """Build a source from validated configuration."""
    if config.source_type == "file":
        if not config.file_path:
            raise ValueError("video.file_path is required when source_type is 'file'")
        return FileVideoSource(config.file_path, capture_factory=capture_factory)
    if config.source_type == "webcam":
        return WebcamVideoSource(config.webcam_index, capture_factory=capture_factory)
    if config.source_type == "rtsp":
        return RTSPVideoSource(
            config.rtsp_url or "",
            read_timeout_s=config.read_timeout_s,
            capture_factory=capture_factory,
        )
    raise ValueError(f"unsupported video source type: {config.source_type!r}")
