"""Headless dashboard view-model.

The dashboard is split into three layers:

* this module decides *what* the user should see, using fusion evidence and the
  Stage 3 signal controller;
* `widgets.py` knows *how* to draw those immutable snapshots with Tkinter;
* `app.py` owns acquisition workers and the main-thread render loop.

Nothing in this module imports Tkinter, OpenCV, Torch or Librosa, so the whole
dashboard logic is testable without a display or a model checkpoint.

Threading contract
------------------
Acquisition workers may call `observe_video`, `observe_audio`, `log_event`,
`update_video_stats` and `update_health` from any thread. All state changes are
guarded by one re-entrant lock. Only the main thread calls `tick`, which evaluates
fusion and advances the signal controller, then renders the returned snapshot.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

from ambulance_system.clock import Clock, RealClock
from ambulance_system.config import AppConfig, VALID_POLICIES
from ambulance_system.events import DetectionEvent
from ambulance_system.fusion import FusionDecision, FusionEngine
from ambulance_system.traffic_signal import SignalController

VALID_LOG_LEVELS = ("INFO", "WARNING", "ERROR")


# ---------------------------------------------------------------------------
# Immutable view types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModalityView:
    """One sensor's latest evidence for one approach."""

    active: bool
    confidence: Optional[float]
    last_seen_at: Optional[float]


@dataclass(frozen=True)
class ApproachView:
    """Everything the UI needs to draw one approach card."""

    name: str
    policy: str
    audio: ModalityView
    video: ModalityView
    fused_active: bool
    fused_confidence: Optional[float]
    fused_reason: str
    phase: str
    colour: str
    priority_started_at: Optional[float]
    last_change_at: float
    signal_reason: str
    video_fps: float
    video_latency_ms: float
    frames_dropped: int
    video_error: Optional[str]
    bbox: Optional[tuple[float, float, float, float]]
    frame_index: Optional[int]


@dataclass(frozen=True)
class HealthView:
    """Measured resource and reliability indicators."""

    cpu_percent: Optional[float]
    system_ram_percent: Optional[float]
    process_ram_mb: Optional[float]
    uptime_s: float
    errors: int


@dataclass(frozen=True)
class EventLogEntry:
    """One user-facing event with a monotonic sequence number."""

    sequence: int
    timestamp: float
    level: str
    message: str
    approach: Optional[str] = None


@dataclass(frozen=True)
class DashboardSnapshot:
    """Complete immutable render input for one UI frame."""

    timestamp: float
    running: bool
    approaches: tuple[ApproachView, ...]
    active_approach: Optional[str]
    fault: bool
    health: HealthView
    events: tuple[EventLogEntry, ...]
    status_message: str
    status_level: str

    def approach(self, name: str) -> ApproachView:
        for item in self.approaches:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass
class _ModalityState:
    confidence: Optional[float] = None
    last_seen_at: Optional[float] = None
    positive_at: Optional[float] = None


@dataclass
class _VideoState(_ModalityState):
    bbox: Optional[tuple[float, float, float, float]] = None
    frame_index: Optional[int] = None


@dataclass
class _VideoStats:
    fps: float = 0.0
    latency_ms: float = 0.0
    frames_dropped: int = 0
    last_error: Optional[str] = None


def _finite_optional(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or None")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _confidence(value: Optional[float], name: str = "confidence") -> Optional[float]:
    result = _finite_optional(value, name)
    if result is not None and not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0]")
    return result


def _non_negative_optional(value: Optional[float], name: str) -> Optional[float]:
    result = _finite_optional(value, name)
    if result is not None and result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


class DashboardViewModel:
    """Fusion + signal state behind an immutable, render-ready snapshot."""

    def __init__(
        self,
        config: AppConfig,
        *,
        clock: Optional[Clock] = None,
        event_history: int = 200,
    ) -> None:
        if isinstance(event_history, bool) or not isinstance(event_history, int) or event_history < 1:
            raise ValueError("event_history must be an int >= 1")

        self._config = config
        self._approaches = tuple(config.system.approach_names)
        self._clock = clock or RealClock()
        self._event_history = event_history

        initial_policy = config.fusion.policy
        self._fusion = FusionEngine(
            config.fusion,
            self._approaches,
            policies={name: initial_policy for name in self._approaches},
            clock=self._clock,
        )
        self._signal = SignalController(
            config.signal,
            self._approaches,
            clock=self._clock,
        )

        self._lock = threading.RLock()
        self._audio = {name: _ModalityState() for name in self._approaches}
        self._video = {name: _VideoState() for name in self._approaches}
        self._video_stats = {name: _VideoStats() for name in self._approaches}
        self._decisions: dict[str, FusionDecision] = {}
        self._events: list[EventLogEntry] = []
        self._next_sequence = 1

        self._cpu_percent: Optional[float] = None
        self._system_ram_percent: Optional[float] = None
        self._process_ram_mb: Optional[float] = None

        self._running = False
        self._started_at = float(self._clock.now())
        self._status_message = "Idle"
        self._status_level = "INFO"
        self._errors = 0

    # ------------------------------------------------------------------
    # Lifecycle and policy
    # ------------------------------------------------------------------
    @property
    def approaches(self) -> tuple[str, ...]:
        return self._approaches

    @property
    def config(self) -> AppConfig:
        return self._config

    def policy_for(self, approach: str) -> str:
        with self._lock:
            return self._fusion.policy_for(approach)

    def start(self) -> DashboardSnapshot:
        with self._lock:
            self._running = True
            self._started_at = float(self._clock.now())
            self._append_event_locked("INFO", "Dashboard started")
            return self._snapshot_locked()

    def stop(self) -> DashboardSnapshot:
        with self._lock:
            self._running = False
            self._append_event_locked("INFO", "Dashboard stopped")
            return self._snapshot_locked()

    def set_policy(self, approach: str, policy: str) -> DashboardSnapshot:
        """Change one approach's policy. Existing evidence is kept."""
        with self._lock:
            if policy not in VALID_POLICIES:
                raise ValueError(f"unknown fusion policy: {policy!r}")
            previous = self._fusion.policy_for(approach)
            self._fusion.set_policy(approach, policy)
            if previous != policy:
                self._append_event_locked(
                    "INFO", f"policy changed from {previous} to {policy}", approach
                )
            return self._snapshot_locked()

    # ------------------------------------------------------------------
    # Evidence ingest (worker threads)
    # ------------------------------------------------------------------
    def observe_video(
        self,
        approach: str,
        confidence: Optional[float],
        *,
        detected: bool,
        bbox: Optional[tuple[float, float, float, float]] = None,
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Record one video frame's outcome and submit positive evidence.

        Returns True when fusion accepted a new event. Duplicates inside the
        debounce window are suppressed by the fusion engine.
        """
        if not isinstance(detected, bool):
            raise ValueError("detected must be a bool")
        value = _confidence(confidence)
        if detected and value is None:
            raise ValueError("confidence is required when detected=True")
        when = float(self._clock.now() if timestamp is None else timestamp)
        self._validate_timestamp(when)

        with self._lock:
            state = self._video[self._require_approach(approach)]
            state.last_seen_at = when
            state.confidence = value
            state.frame_index = frame_index
            if not detected:
                state.positive_at = None
                state.bbox = None
                return False

            clean_bbox = tuple(float(item) for item in bbox) if bbox is not None else None
            state.positive_at = when
            state.bbox = clean_bbox  # type: ignore[assignment]
            event = DetectionEvent.video(
                approach=approach,
                confidence=value,
                bbox=clean_bbox,
                frame_index=frame_index,
                timestamp=when,
            )
            return self._fusion.submit(event)

    def observe_audio(
        self,
        approach: str,
        confidence: Optional[float],
        *,
        is_siren: bool,
        window_start_s: Optional[float] = None,
        window_end_s: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Record one classified audio window and submit positive evidence."""
        if not isinstance(is_siren, bool):
            raise ValueError("is_siren must be a bool")
        value = _confidence(confidence)
        if is_siren and value is None:
            value = 1.0  # mirrors WindowPrediction.to_event for estimators without proba
        when = float(self._clock.now() if timestamp is None else timestamp)
        self._validate_timestamp(when)

        with self._lock:
            state = self._audio[self._require_approach(approach)]
            state.last_seen_at = when
            state.confidence = value
            if not is_siren:
                state.positive_at = None
                return False

            state.positive_at = when
            event = DetectionEvent.audio(
                approach=approach,
                confidence=value,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                timestamp=when,
            )
            return self._fusion.submit(event)

    # ------------------------------------------------------------------
    # Decision cycle and rendering snapshot
    # ------------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> DashboardSnapshot:
        """Evaluate fusion, advance the signal controller and return a snapshot."""
        when = float(self._clock.now() if now is None else now)
        self._validate_timestamp(when)

        with self._lock:
            update = self._fusion.evaluate(when)
            self._decisions.update(update.decisions)

            for decision in update.changed:
                if decision.active:
                    self._append_event_locked(
                        "INFO",
                        f"emergency evidence active: {decision.reason}",
                        decision.approach,
                    )
                else:
                    self._append_event_locked(
                        "INFO",
                        f"emergency evidence cleared: {decision.reason}",
                        decision.approach,
                    )

            commands = self._signal.step(update.to_observations(), now=when)
            for command in commands:
                self._append_event_locked(
                    "INFO",
                    f"signal {command.colour.value.upper()} in {command.phase.value}: {command.reason}",
                    command.approach,
                )

            return self._snapshot_locked(when)

    def snapshot(self) -> DashboardSnapshot:
        """Return the current state without running a decision cycle."""
        with self._lock:
            return self._snapshot_locked()

    # ------------------------------------------------------------------
    # Metrics and health (worker or main thread)
    # ------------------------------------------------------------------
    def update_video_stats(
        self,
        approach: str,
        *,
        fps: Optional[float] = None,
        latency_ms: Optional[float] = None,
        frames_dropped: Optional[int] = None,
        last_error: Optional[str] = None,
    ) -> None:
        with self._lock:
            stats = self._video_stats[self._require_approach(approach)]
            if fps is not None:
                stats.fps = _non_negative_optional(fps, "fps") or 0.0
            if latency_ms is not None:
                stats.latency_ms = _non_negative_optional(latency_ms, "latency_ms") or 0.0
            if frames_dropped is not None:
                if isinstance(frames_dropped, bool) or not isinstance(frames_dropped, int) or frames_dropped < 0:
                    raise ValueError("frames_dropped must be an int >= 0")
                stats.frames_dropped = frames_dropped
            stats.last_error = last_error

    def update_health(
        self,
        *,
        cpu_percent: Optional[float] = None,
        system_ram_percent: Optional[float] = None,
        process_ram_mb: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._cpu_percent = _non_negative_optional(cpu_percent, "cpu_percent")
            self._system_ram_percent = _non_negative_optional(
                system_ram_percent, "system_ram_percent"
            )
            self._process_ram_mb = _non_negative_optional(process_ram_mb, "process_ram_mb")

    # ------------------------------------------------------------------
    # User actions and notifications
    # ------------------------------------------------------------------
    def log_event(self, level: str, message: str, approach: Optional[str] = None) -> None:
        """Append a notification from any thread."""
        with self._lock:
            self._append_event_locked(level, message, approach)

    def clear_all(self, reason: str = "manual Clear All") -> DashboardSnapshot:
        """Legacy Clear All: drop evidence, reset signals and restore policies."""
        when = float(self._clock.now())
        with self._lock:
            self._fusion.reset()
            self._signal.clear_all(now=when)
            for name in self._approaches:
                self._audio[name] = _ModalityState()
                self._video[name] = _VideoState()
                self._video_stats[name] = _VideoStats()
                self._fusion.set_policy(name, self._config.fusion.policy)
            self._decisions.clear()
            self._append_event_locked("INFO", reason)
            self._status_message = reason
            self._status_level = "INFO"
            return self._snapshot_locked(when)

    def trigger_fault(self, reason: str = "manual fault simulation") -> DashboardSnapshot:
        """Force the simulated controller into all-red FAULT for demos/tests."""
        when = float(self._clock.now())
        with self._lock:
            commands = self._signal.trigger_fault(reason, now=when)
            for command in commands:
                self._append_event_locked(
                    "ERROR",
                    f"signal {command.colour.value.upper()} in {command.phase.value}: {command.reason}",
                    command.approach,
                )
            return self._snapshot_locked(when)

    def clear_fault(self, reason: str = "fault acknowledged") -> DashboardSnapshot:
        when = float(self._clock.now())
        with self._lock:
            if self._signal.is_fault:
                self._signal.clear_fault(reason, now=when)
                self._append_event_locked("INFO", reason)
            return self._snapshot_locked(when)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_approach(self, approach: str) -> str:
        if approach not in self._audio:
            raise ValueError(f"unknown approach: {approach!r}")
        return approach

    @staticmethod
    def _validate_timestamp(value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timestamp must be a number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("timestamp must be finite and >= 0")

    def _append_event_locked(self, level: str, message: str, approach: Optional[str] = None) -> None:
        level = str(level).upper()
        if level not in VALID_LOG_LEVELS:
            raise ValueError(f"level must be one of {VALID_LOG_LEVELS}, got {level!r}")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if approach is not None:
            self._require_approach(approach)

        now = float(self._clock.now())
        entry = EventLogEntry(
            sequence=self._next_sequence,
            timestamp=now,
            level=level,
            message=message.strip(),
            approach=approach,
        )
        self._next_sequence += 1
        self._events.append(entry)
        if len(self._events) > self._event_history:
            del self._events[: len(self._events) - self._event_history]

        self._status_message = entry.message
        self._status_level = level
        if level == "ERROR":
            self._errors += 1

    def _snapshot_locked(self, now: Optional[float] = None) -> DashboardSnapshot:
        when = float(self._clock.now() if now is None else now)
        states = self._signal.states
        stale = float(self._config.fusion.stale_after_s)
        views: list[ApproachView] = []

        for name in self._approaches:
            signal_state = states[name]
            decision = self._decisions.get(name)
            audio = self._audio[name]
            video = self._video[name]
            stats = self._video_stats[name]

            audio_active = audio.positive_at is not None and when - audio.positive_at <= stale
            video_active = video.positive_at is not None and when - video.positive_at <= stale

            views.append(
                ApproachView(
                    name=name,
                    policy=self._fusion.policy_for(name),
                    audio=ModalityView(audio_active, audio.confidence, audio.last_seen_at),
                    video=ModalityView(video_active, video.confidence, video.last_seen_at),
                    fused_active=False if decision is None else decision.active,
                    fused_confidence=None if decision is None else decision.confidence,
                    fused_reason="waiting for evaluation" if decision is None else decision.reason,
                    phase=signal_state.phase.value,
                    colour=signal_state.colour.value,
                    priority_started_at=signal_state.priority_started_at,
                    last_change_at=signal_state.last_change_at,
                    signal_reason=signal_state.reason,
                    video_fps=stats.fps,
                    video_latency_ms=stats.latency_ms,
                    frames_dropped=stats.frames_dropped,
                    video_error=stats.last_error,
                    bbox=video.bbox if video_active else None,
                    frame_index=video.frame_index if video_active else None,
                )
            )

        uptime = max(0.0, when - self._started_at) if self._running else 0.0
        return DashboardSnapshot(
            timestamp=when,
            running=self._running,
            approaches=tuple(views),
            active_approach=self._signal.active_approach,
            fault=self._signal.is_fault,
            health=HealthView(
                cpu_percent=self._cpu_percent,
                system_ram_percent=self._system_ram_percent,
                process_ram_mb=self._process_ram_mb,
                uptime_s=uptime,
                errors=self._errors,
            ),
            events=tuple(self._events),
            status_message=self._status_message,
            status_level=self._status_level,
        )
