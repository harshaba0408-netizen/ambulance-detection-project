"""Acquisition + processing pipeline with a bounded queue.

The acquisition thread reads frames and drops them when the processing worker
cannot keep up. The processing worker invokes the caller's callback for every
delivered frame; `packet.sampled` indicates whether this frame is due for
inference according to `frame_skip`. This keeps preview latency low while
running detection on a subset of frames.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ambulance_system.clock import Clock, RealClock
from ambulance_system.logging_utils import get_logger
from ambulance_system.monitoring.metrics import LatencyTracker
from ambulance_system.video.source import VideoSource

_SENTINEL = object()


@dataclass(frozen=True)
class FramePacket:
    """One delivered frame plus the metadata the consumers need."""

    index: int
    frame: np.ndarray
    sampled: bool
    timestamp: float

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative int")
        if not isinstance(self.frame, np.ndarray):
            raise ValueError("frame must be a numpy array")
        if not isinstance(self.sampled, bool):
            raise ValueError("sampled must be a bool")
        if self.timestamp < 0:
            raise ValueError("timestamp must be >= 0")


@dataclass(frozen=True)
class PipelineStats:
    """Snapshot of pipeline counters and measured rates."""

    frames_read: int
    frames_delivered: int
    frames_dropped: int
    samples_processed: int
    errors: int
    reconnects: int
    fps: float
    avg_latency_ms: float
    p95_latency_ms: float
    running: bool
    last_error: Optional[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "frames_read": self.frames_read,
            "frames_delivered": self.frames_delivered,
            "frames_dropped": self.frames_dropped,
            "samples_processed": self.samples_processed,
            "errors": self.errors,
            "reconnects": self.reconnects,
            "fps": self.fps,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "running": self.running,
            "last_error": self.last_error,
        }


class VideoPipeline:
    """Two-thread video pipeline with sampling and reconnection."""

    def __init__(
        self,
        source: VideoSource,
        on_frame,
        *,
        frame_skip: int = 5,
        queue_size: int = 2,
        read_delay_s: float = 0.0,
        reconnect_delay_s: float = 2.0,
        max_reconnect_attempts: Optional[int] = None,
        clock: Optional[Clock] = None,
        logger=None,
    ) -> None:
        if not callable(on_frame):
            raise TypeError("on_frame must be callable")
        if isinstance(frame_skip, bool) or not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be an int >= 1")
        if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 1:
            raise ValueError("queue_size must be an int >= 1")
        if read_delay_s < 0 or reconnect_delay_s < 0:
            raise ValueError("delays must be >= 0")
        if max_reconnect_attempts is not None and max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts must be >= 0 or None")

        self.source = source
        self._on_frame = on_frame
        self._frame_skip = frame_skip
        self._read_delay_s = float(read_delay_s)
        self._reconnect_delay_s = float(reconnect_delay_s)
        self._max_reconnect_attempts = max_reconnect_attempts
        self._clock = clock or RealClock()
        self._logger = logger or get_logger("video")

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._started_at = 0.0
        self._latency = LatencyTracker()

        self._frames_read = 0
        self._frames_delivered = 0
        self._frames_dropped = 0
        self._samples_processed = 0
        self._errors = 0
        self._reconnects = 0
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started and self._any_thread_alive():
                raise RuntimeError("pipeline is already running")
            self._stop_event.clear()
            self._started = True
            self._started_at = self._clock.monotonic()
            self._frames_read = 0
            self._frames_delivered = 0
            self._frames_dropped = 0
            self._samples_processed = 0
            self._errors = 0
            self._reconnects = 0
            self._last_error = None
            self._latency.reset()
            self._drain_queue()

        try:
            self.source.open()
        except Exception:
            with self._lock:
                self._started = False
            raise

        self._threads = [
            threading.Thread(target=self._acquire_loop, daemon=True, name="video-acquire"),
            threading.Thread(target=self._process_loop, daemon=True, name="video-process"),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        try:
            self.source.close()
        except Exception:
            pass

        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self._threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(timeout=remaining)

        with self._lock:
            self._started = self._any_thread_alive()

    def wait(self, timeout: float = 2.0) -> bool:
        """Wait for both workers to finish. Returns True if they did."""
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self._threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(timeout=remaining)
        return not self._any_thread_alive()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._started and self._any_thread_alive()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @property
    def stats(self) -> PipelineStats:
        with self._lock:
            elapsed = max(self._clock.monotonic() - self._started_at, 1e-9)
            fps = self._frames_delivered / elapsed if self._started else 0.0
            return PipelineStats(
                frames_read=self._frames_read,
                frames_delivered=self._frames_delivered,
                frames_dropped=self._frames_dropped,
                samples_processed=self._samples_processed,
                errors=self._errors,
                reconnects=self._reconnects,
                fps=fps,
                avg_latency_ms=self._latency.average_ms,
                p95_latency_ms=self._latency.p95_ms,
                running=self._started and self._any_thread_alive(),
                last_error=self._last_error,
            )

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def _acquire_loop(self) -> None:
        reconnect_attempts = 0
        try:
            while not self._stop_event.is_set():
                frame = self.source.read()

                if frame is None:
                    if self.source.exhausted:
                        break
                    if (
                        self._max_reconnect_attempts is not None
                        and reconnect_attempts >= self._max_reconnect_attempts
                    ):
                        self._last_error = f"reconnect limit reached for {self.source.name}"
                        self._logger.warning(self._last_error)
                        break

                    reconnect_attempts += 1
                    with self._lock:
                        self._reconnects += 1
                    self._logger.warning(
                        "video source read failed; reconnecting",
                        extra={"event": {"source": self.source.name, "attempt": reconnect_attempts}},
                    )
                    try:
                        self.source.close()
                        if self._stop_event.wait(self._reconnect_delay_s):
                            break
                        self.source.open()
                    except Exception as exc:
                        self._last_error = str(exc)
                        self._logger.warning("video reconnect failed: %s", exc)
                        if self._stop_event.wait(self._reconnect_delay_s):
                            break
                    continue

                reconnect_attempts = 0
                with self._lock:
                    index = self._frames_read
                    self._frames_read += 1

                packet = FramePacket(
                    index=index,
                    frame=frame,
                    sampled=(index % self._frame_skip == 0),
                    timestamp=self._clock.now(),
                )
                try:
                    self._queue.put(packet, timeout=0.1)
                except queue.Full:
                    with self._lock:
                        self._frames_dropped += 1

                if self._read_delay_s and self._stop_event.wait(self._read_delay_s):
                    break
        except Exception as exc:
            self._last_error = str(exc)
            self._logger.exception("video acquisition failed")
        finally:
            self._send_sentinel()

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break

            packet: FramePacket = item
            with self._lock:
                self._frames_delivered += 1
                if packet.sampled:
                    self._samples_processed += 1

            started = self._clock.monotonic()
            try:
                self._on_frame(packet)
            except Exception:
                with self._lock:
                    self._errors += 1
                self._logger.exception("video frame callback failed")
            finally:
                self._latency.add(self._clock.monotonic() - started)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _any_thread_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def _send_sentinel(self) -> None:
        # Never discard a pending frame just to deliver the sentinel: the
        # processing worker is still consuming during a normal end-of-stream.
        while not self._stop_event.is_set():
            try:
                self._queue.put(_SENTINEL, timeout=0.05)
                return
            except queue.Full:
                continue

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
