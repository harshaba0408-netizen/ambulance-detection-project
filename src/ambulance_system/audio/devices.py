"""Audio sources.

`FileAudioSource` decodes the whole file once (the published clips are about
3 seconds) and then yields fixed-size blocks. `MicrophoneAudioSource` captures
the same fixed-size blocks from a live input device. Both implement the same
`AudioSource` protocol, so windowing and classification are shared.

`sounddevice` is imported lazily by the microphone source and by
`list_input_devices`, so the rest of the package still imports and its tests run
on a machine where the optional backend is absent.

`FileAudioSource` keeps the whole decoded clip in memory deliberately; it does
not pretend to be a low-latency device.
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np

from ambulance_system.audio.features import load_audio


class AudioSourceError(RuntimeError):
    """Raised when an audio source cannot be opened or read."""


@runtime_checkable
class AudioSource(Protocol):
    """Minimal mono block source interface."""

    @property
    def name(self) -> str:
        ...

    @property
    def sample_rate(self) -> int:
        ...

    @property
    def is_open(self) -> bool:
        ...

    @property
    def exhausted(self) -> bool:
        ...

    def open(self) -> None:
        ...

    def read(self) -> Optional[np.ndarray]:
        ...

    def close(self) -> None:
        ...


class FileAudioSource:
    """Yield a recorded file as fixed-size mono blocks."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = 22050,
        block_s: float = 0.25,
        duration: Optional[float] = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"audio file not found: {self.path}")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive int")
        if (
            isinstance(block_s, bool)
            or not isinstance(block_s, (int, float))
            or not math.isfinite(float(block_s))
            or float(block_s) <= 0
        ):
            raise ValueError("block_s must be > 0")
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError("duration must be > 0 or None")

        self._sample_rate = sample_rate
        self._block_samples = max(1, int(round(float(block_s) * sample_rate)))
        self._duration = duration
        self._samples: Optional[np.ndarray] = None
        self._offset = 0
        self._exhausted = False

    @property
    def name(self) -> str:
        return f"file:{self.path.name}"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def block_samples(self) -> int:
        return self._block_samples

    @property
    def is_open(self) -> bool:
        return self._samples is not None

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def open(self) -> None:
        if self._samples is not None:
            return
        try:
            samples, rate = load_audio(
                self.path,
                sample_rate=self._sample_rate,
                duration=self._duration,
            )
        except Exception as exc:
            raise AudioSourceError(f"could not load audio source {self.name}: {exc}") from exc
        if rate != self._sample_rate:
            raise AudioSourceError(
                f"expected {self._sample_rate} Hz from {self.name}, got {rate} Hz"
            )
        self._samples = samples
        self._offset = 0
        self._exhausted = False

    def read(self) -> Optional[np.ndarray]:
        if self._samples is None:
            return None
        if self._offset >= self._samples.size:
            self._exhausted = True
            return None
        end = min(self._offset + self._block_samples, self._samples.size)
        block = self._samples[self._offset : end]
        self._offset = end
        return block

    def close(self) -> None:
        self._samples = None
        self._offset = 0


# ---------------------------------------------------------------------------
# Live microphone source
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioDeviceInfo:
    """One input-capable device reported by the audio backend."""

    index: int
    name: str
    max_input_channels: int
    default_samplerate: float

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative int")
        if not isinstance(self.name, str):
            raise ValueError("name must be a str")
        if (
            isinstance(self.max_input_channels, bool)
            or not isinstance(self.max_input_channels, int)
            or self.max_input_channels <= 0
        ):
            raise ValueError("max_input_channels must be a positive int")
        if (
            isinstance(self.default_samplerate, bool)
            or not isinstance(self.default_samplerate, (int, float))
            or not math.isfinite(float(self.default_samplerate))
            or float(self.default_samplerate) <= 0
        ):
            raise ValueError("default_samplerate must be > 0")


class _InputStream(Protocol):
    """The slice of the sounddevice stream API this module relies on."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def close(self) -> None:
        ...


def _load_sounddevice() -> Any:
    try:
        import sounddevice  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise AudioSourceError(
            "live microphone capture requires the optional 'sounddevice' package; "
            "install it with `pip install sounddevice`"
        ) from exc
    return sounddevice


def _default_stream_factory(
    *,
    device: Optional[str | int],
    sample_rate: int,
    block_samples: int,
    channels: int,
    callback: Callable[..., None],
) -> _InputStream:
    sounddevice = _load_sounddevice()
    return sounddevice.InputStream(
        device=device,
        samplerate=sample_rate,
        blocksize=block_samples,
        channels=channels,
        dtype="float32",
        callback=callback,
    )


def list_input_devices(
    device_query: Optional[Callable[[], Any]] = None,
) -> tuple[AudioDeviceInfo, ...]:
    """Return input-capable devices, hiding output-only entries.

    `device_query` is injectable so the formatting logic can be tested without
    the optional backend or real hardware.
    """
    query = device_query if device_query is not None else _load_sounddevice().query_devices
    try:
        raw_devices = query()
    except AudioSourceError:
        raise
    except Exception as exc:
        raise AudioSourceError(f"could not list audio devices: {exc}") from exc

    devices: list[AudioDeviceInfo] = []
    for index, entry in enumerate(raw_devices):
        channels = int(entry.get("max_input_channels", 0))
        if channels <= 0:
            continue
        devices.append(
            AudioDeviceInfo(
                index=index,
                name=str(entry.get("name", "")),
                max_input_channels=channels,
                default_samplerate=float(entry.get("default_samplerate", 0.0)),
            )
        )
    return tuple(devices)


_STOP = object()

#: Nominal float32 audio lies in [-1, 1]; anything far outside is a driver or
#: device fault (seen with a blocked Windows microphone returning uninitialised
#: buffers), not sound.
MAX_ABS_SAMPLE = 4.0


class MicrophoneAudioSource:
    """Capture fixed-size mono blocks from a live input device.

    The sounddevice callback runs on a PortAudio thread, so blocks are handed to
    `read()` through a bounded queue. If the consumer falls behind, the newest
    block is dropped and counted in `dropped_blocks` rather than blocking the
    audio thread. Blocks that are empty, non-finite, or outside
    `MAX_ABS_SAMPLE` are rejected and counted in `invalid_blocks`; this keeps a
    faulty device from feeding garbage to the classifier. `read()` raises
    `AudioSourceError` if the backend stalls for longer than `read_timeout_s`;
    a silent microphone still delivers zero-valued blocks, so a stall is a real
    fault, not silence.
    """

    def __init__(
        self,
        *,
        device: Optional[str | int] = None,
        sample_rate: int = 22050,
        block_s: float = 0.25,
        channels: int = 1,
        buffer_blocks: int = 8,
        read_timeout_s: float = 2.0,
        stream_factory: Optional[Callable[..., _InputStream]] = None,
    ) -> None:
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive int")
        if (
            isinstance(block_s, bool)
            or not isinstance(block_s, (int, float))
            or not math.isfinite(float(block_s))
            or float(block_s) <= 0
        ):
            raise ValueError("block_s must be > 0")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if (
            isinstance(buffer_blocks, bool)
            or not isinstance(buffer_blocks, int)
            or buffer_blocks <= 0
        ):
            raise ValueError("buffer_blocks must be a positive int")
        if (
            isinstance(read_timeout_s, bool)
            or not isinstance(read_timeout_s, (int, float))
            or not math.isfinite(float(read_timeout_s))
            or float(read_timeout_s) <= 0
        ):
            raise ValueError("read_timeout_s must be > 0")
        if device is not None and not isinstance(device, (str, int)):
            raise ValueError("device must be a str, an int or None")
        if stream_factory is not None and not callable(stream_factory):
            raise ValueError("stream_factory must be callable or None")

        self._device = device
        self._sample_rate = sample_rate
        self._block_samples = max(1, int(round(float(block_s) * sample_rate)))
        self._channels = channels
        self._buffer_blocks = buffer_blocks
        self._read_timeout_s = float(read_timeout_s)
        self._stream_factory = stream_factory
        self._stream: Optional[_InputStream] = None
        self._queue: Optional[queue.Queue[Any]] = None
        self._exhausted = False
        self._dropped = 0
        self._invalid = 0
        self._last_status: Optional[str] = None

    @property
    def name(self) -> str:
        return "mic:default" if self._device is None else f"mic:{self._device}"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def block_samples(self) -> int:
        return self._block_samples

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def dropped_blocks(self) -> int:
        return self._dropped

    @property
    def invalid_blocks(self) -> int:
        return self._invalid

    @property
    def last_status(self) -> Optional[str]:
        return self._last_status

    def open(self) -> None:
        if self._stream is not None:
            return
        factory = self._stream_factory or _default_stream_factory
        self._queue = queue.Queue(maxsize=self._buffer_blocks)
        self._dropped = 0
        self._invalid = 0
        self._last_status = None
        self._exhausted = False
        stream: Optional[_InputStream] = None
        try:
            stream = factory(
                device=self._device,
                sample_rate=self._sample_rate,
                block_samples=self._block_samples,
                channels=self._channels,
                callback=self._on_audio,
            )
            stream.start()
        except AudioSourceError:
            raise
        except Exception as exc:
            if stream is not None:
                self._close_stream(stream)
            self._queue = None
            raise AudioSourceError(
                f"could not open microphone source {self.name}: {exc}"
            ) from exc
        self._stream = stream

    def read(self) -> Optional[np.ndarray]:
        if self._exhausted or self._stream is None:
            return None
        audio_queue = self._queue
        if audio_queue is None:  # defensive: open() always sets it
            return None
        try:
            block = audio_queue.get(timeout=self._read_timeout_s)
        except queue.Empty:
            if self._exhausted:
                return None
            detail = (
                f"; rejected {self._invalid} invalid block(s)"
                if self._invalid
                else ""
            )
            raise AudioSourceError(
                f"no audio received from {self.name} within "
                f"{self._read_timeout_s:.2f}s{detail}"
            )
        if block is _STOP or self._exhausted:
            return None
        return block

    def close(self) -> None:
        stream, self._stream = self._stream, None
        self._exhausted = True
        if stream is not None:
            self._close_stream(stream)
        audio_queue = self._queue
        if audio_queue is not None:
            try:
                audio_queue.put_nowait(_STOP)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                    audio_queue.put_nowait(_STOP)
                except (queue.Empty, queue.Full):
                    pass

    # ------------------------------------------------------------------
    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any = None,
        status: Any = None,
    ) -> None:
        if status is not None:
            self._last_status = str(status)
        block = np.asarray(indata, dtype=np.float32)
        if block.ndim == 2:
            block = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
        elif block.ndim != 1:
            return
        # PortAudio owns this buffer and reuses it after the callback returns,
        # so the queue must never hold a reference to it. `ascontiguousarray`
        # is a no-op for an already contiguous float32 array, hence the forced
        # copy here.
        block = np.array(block, dtype=np.float32, order="C", copy=True)
        if (
            block.size == 0
            or not np.all(np.isfinite(block))
            or float(np.max(np.abs(block))) > MAX_ABS_SAMPLE
        ):
            self._invalid += 1
            return
        audio_queue = self._queue
        if audio_queue is None:
            return
        try:
            audio_queue.put_nowait(block)
        except queue.Full:
            self._dropped += 1

    @staticmethod
    def _close_stream(stream: _InputStream) -> None:
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
