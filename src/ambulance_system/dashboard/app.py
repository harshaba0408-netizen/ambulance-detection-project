"""Tkinter application shell (Stage 8).

Responsibilities are deliberately narrow:

* own the acquisition workers (video pipeline, audio stream),
* hand evidence and errors to `DashboardViewModel` from worker threads,
* render immutable snapshots on the Tk main thread via `after()`,
* translate buttons into view-model commands.

There are no Tk widget calls from worker threads and no business logic in this
module. `python new.py` remains the launch command.
"""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ambulance_system.audio import FileAudioSource, SirenClassifier, stream_predictions
from ambulance_system.config import AppConfig, load_config
from ambulance_system.dashboard.viewmodel import DashboardSnapshot, DashboardViewModel
from ambulance_system.dashboard.widgets import ApproachCard, EventLogPanel, HealthPanel
from ambulance_system.detection import Detector
from ambulance_system.logging_utils import configure_logging
from ambulance_system.monitoring.metrics import ResourceMonitor
from ambulance_system.video import FileVideoSource, VideoPipeline

WINDOW_TITLE = "AI-Driven Ambulance Traffic Decongestion Dashboard"
REFRESH_INTERVAL_MS = 100
BACKGROUND = "#233047"
TEXT_COLOR = "white"


class FrameStore:
    """Latest resized RGB frame per approach, written by workers."""

    def __init__(self, display_size: tuple[int, int]) -> None:
        self._width, self._height = display_size
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[np.ndarray, Optional[tuple[float, float, float, float]]]] = {}

    def put(
        self,
        approach: str,
        frame: np.ndarray,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        if frame.ndim != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
            return
        source_h, source_w = frame.shape[:2]
        channels = frame.shape[2] if frame.ndim == 3 else 0
        if channels == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif channels == 1:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self._width, self._height), interpolation=cv2.INTER_AREA)

        scaled_box = None
        if bbox is not None:
            scale_x = self._width / float(source_w)
            scale_y = self._height / float(source_h)
            x1, y1, x2, y2 = (float(value) for value in bbox)
            scaled_box = (
                max(0.0, min(self._width - 1.0, x1 * scale_x)),
                max(0.0, min(self._height - 1.0, y1 * scale_y)),
                max(0.0, min(self._width - 1.0, x2 * scale_x)),
                max(0.0, min(self._height - 1.0, y2 * scale_y)),
            )

        with self._lock:
            self._frames[approach] = (rgb, scaled_box)

    def get(
        self, approach: str
    ) -> Optional[tuple[np.ndarray, Optional[tuple[float, float, float, float]]]]:
        with self._lock:
            return self._frames.get(approach)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


class DashboardApp:
    """Main-thread Tk shell around a `DashboardViewModel`."""

    def __init__(
        self,
        root: tk.Tk,
        config: AppConfig,
        *,
        detector: Optional[Detector] = None,
        classifier: Optional[SirenClassifier] = None,
        viewmodel: Optional[DashboardViewModel] = None,
    ) -> None:
        self.root = root
        self.config = config
        self.viewmodel = viewmodel or DashboardViewModel(config)
        self._clock = time.monotonic
        self._closing = False

        self.detector = detector or self._load_detector()
        self.classifier = classifier or self._load_classifier()

        self.files: dict[str, dict[str, str]] = {
            name: {} for name in self.viewmodel.approaches
        }
        self.pipelines: dict[str, VideoPipeline] = {}
        self._audio_runners: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self.frames = FrameStore(
            (config.video.display_width, config.video.display_height)
        )
        self._resources = ResourceMonitor()
        self._last_health_sample = self._clock()

        self._cards: dict[str, ApproachCard] = {}
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.viewmodel.start()
        self.root.after(REFRESH_INTERVAL_MS, self.refresh)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_detector(self) -> Detector:
        return Detector(
            self.config.model.yolo_path,
            confidence_threshold=self.config.model.confidence_threshold,
            imgsz=self.config.model.imgsz,
        )

    def _load_classifier(self) -> SirenClassifier:
        path = self.config.model.siren_classifier_path
        if not Path(path).is_file():
            raise FileNotFoundError(f"Siren classifier missing: {path}")
        return SirenClassifier.from_path(
            path,
            threshold=self.config.audio.threshold,
            sample_rate=self.config.audio.sample_rate,
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1220x860")
        self.root.configure(bg=BACKGROUND)

        tk.Label(
            self.root,
            text="Ambulance Traffic Decongestion Dashboard",
            font=("Helvetica", 20, "bold"),
            bg=BACKGROUND,
            fg=TEXT_COLOR,
        ).pack(pady=(8, 2))

        self.status_label = tk.Label(
            self.root,
            text="Idle",
            font=("Helvetica", 11),
            bg=BACKGROUND,
            fg=TEXT_COLOR,
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, padx=12)

        cards_frame = tk.Frame(self.root, bg=BACKGROUND)
        cards_frame.pack(pady=4)
        for index, name in enumerate(self.viewmodel.approaches):
            card = ApproachCard(
                cards_frame,
                name,
                video_size=(
                    self.config.video.display_width,
                    self.config.video.display_height,
                ),
                on_select_video=lambda i=index: self.select_file(i, "video"),
                on_select_audio=lambda i=index: self.select_file(i, "audio"),
                on_start=lambda i=index: self.start_approach(i),
                on_stop=lambda i=index: self.stop_approach(i),
                on_policy_change=lambda value, i=index: self.change_policy(i, value),
            )
            card.grid(row=index // 2, column=index % 2, padx=8, pady=6, sticky="n")
            self._cards[name] = card

        bottom = tk.Frame(self.root, bg=BACKGROUND)
        bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        self.health_panel = HealthPanel(bottom)
        self.health_panel.pack(side=tk.LEFT, anchor="n", padx=(0, 12))

        buttons = tk.Frame(bottom, bg=BACKGROUND)
        buttons.pack(side=tk.LEFT, anchor="n", padx=(0, 12))
        ttk.Button(buttons, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=2)
        ttk.Button(buttons, text="Clear Fault", command=self.clear_fault).pack(fill=tk.X, pady=2)
        ttk.Button(buttons, text="Simulate Fault", command=self.simulate_fault).pack(
            fill=tk.X, pady=2
        )

        self.event_log = EventLogPanel(bottom)
        self.event_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # File selection and policy
    # ------------------------------------------------------------------
    def select_file(self, index: int, kind: str) -> None:
        filetypes = (
            [("Video files", "*.mp4 *.avi *.mov *.mkv")]
            if kind == "video"
            else [("Audio files", "*.wav *.mp3")]
        )
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return

        name = self.viewmodel.approaches[index]
        self.files[name][kind] = path
        self.viewmodel.log_event(
            "INFO", f"selected {kind}: {os.path.basename(path)}", name
        )
        self._sync_sources(name)
        self._render_now()

    def change_policy(self, index: int, policy: str) -> None:
        name = self.viewmodel.approaches[index]
        self.viewmodel.set_policy(name, policy)
        self._sync_sources(name)
        self._render_now()

    def start_approach(self, index: int) -> None:
        name = self.viewmodel.approaches[index]
        if not self.files[name]:
            self.viewmodel.log_event("WARNING", "no video or audio file selected", name)
            return
        self._sync_sources(name)
        self._render_now()

    def stop_approach(self, index: int) -> None:
        name = self.viewmodel.approaches[index]
        self._stop_video(name)
        self._stop_audio(name)
        self.viewmodel.log_event("INFO", "sources stopped", name)
        self._render_now()

    def _sync_sources(self, name: str) -> None:
        policy = self.viewmodel.policy_for(name)
        selection = self.files[name]

        if policy == "AUDIO":
            self._stop_video(name)
        if policy == "VIDEO":
            self._stop_audio(name)

        if policy in ("AUDIO", "BOTH", "ADAPTIVE") and "audio" in selection:
            self._start_audio(name, selection["audio"])
        if policy in ("VIDEO", "BOTH", "ADAPTIVE") and "video" in selection:
            self._start_video(name, selection["video"])

        if policy in ("BOTH", "ADAPTIVE"):
            missing = [kind for kind in ("video", "audio") if kind not in selection]
            if missing:
                self.viewmodel.log_event(
                    "WARNING",
                    f"{policy} policy is active but no {', '.join(missing)} file is selected",
                    name,
                )

    # ------------------------------------------------------------------
    # Video worker
    # ------------------------------------------------------------------
    def _start_video(self, name: str, path: str) -> None:
        running = self.pipelines.get(name)
        if running is not None and running.running:
            return
        self._stop_video(name)

        try:
            source = FileVideoSource(path)
        except FileNotFoundError:
            self.viewmodel.log_event("ERROR", f"video file not found: {path}", name)
            return

        last_bbox: list[Optional[tuple[float, float, float, float]]] = [None]

        def on_frame(packet) -> None:
            detection = None
            if packet.sampled:
                try:
                    detection = self.detector.detect_best(
                        packet.frame,
                        frame_index=packet.index,
                        timestamp=packet.timestamp,
                    )
                except Exception as exc:  # pragma: no cover - hardware/model dependent
                    self.viewmodel.log_event("ERROR", f"YOLO error: {exc}", name)

                confidence = None if detection is None else detection.confidence
                bbox = None if detection is None else detection.bbox
                last_bbox[0] = bbox
                self.viewmodel.observe_video(
                    name,
                    confidence,
                    detected=detection is not None,
                    bbox=bbox,
                    frame_index=packet.index,
                    timestamp=packet.timestamp,
                )

            self.frames.put(name, packet.frame, last_bbox[0])

        pipeline = VideoPipeline(
            source,
            on_frame,
            frame_skip=self.config.video.frame_skip,
            read_delay_s=self.config.video.read_delay_s,
            reconnect_delay_s=self.config.video.reconnect_delay_s,
        )
        try:
            pipeline.start()
        except Exception as exc:
            self.viewmodel.log_event("ERROR", f"could not open video: {exc}", name)
            return

        self.pipelines[name] = pipeline
        self.viewmodel.log_event("INFO", f"video started: {os.path.basename(path)}", name)

    def _stop_video(self, name: str) -> None:
        pipeline = self.pipelines.pop(name, None)
        if pipeline is not None:
            pipeline.stop(timeout=0.3)

    # ------------------------------------------------------------------
    # Audio worker
    # ------------------------------------------------------------------
    def _start_audio(self, name: str, path: str) -> None:
        if name in self._audio_runners:
            return

        stop_event = threading.Event()

        def worker() -> None:
            source = None
            predictions = None
            windows = 0
            positives = 0
            try:
                source = FileAudioSource(
                    path,
                    sample_rate=self.config.audio.sample_rate,
                    block_s=self.config.audio.block_s,
                )
                predictions = stream_predictions(
                    source,
                    self.classifier,
                    window_s=self.config.audio.window_s,
                    overlap_s=self.config.audio.overlap_s,
                )
                for prediction in predictions:
                    if stop_event.is_set():
                        break
                    windows += 1
                    if prediction.is_siren:
                        positives += 1
                    self.viewmodel.observe_audio(
                        name,
                        prediction.confidence,
                        is_siren=prediction.is_siren,
                        window_start_s=prediction.window.start_s,
                        window_end_s=prediction.window.end_s,
                    )
                if not stop_event.is_set():
                    self.viewmodel.log_event(
                        "INFO",
                        f"audio finished: {windows} windows, {positives} siren",
                        name,
                    )
            except Exception as exc:  # pragma: no cover - file/model dependent
                self.viewmodel.log_event("ERROR", f"audio error: {exc}", name)
            finally:
                try:
                    if predictions is not None:
                        predictions.close()
                except Exception:
                    pass
                if source is not None:
                    source.close()
                current = self._audio_runners.get(name)
                if current is not None and current[1] is stop_event:
                    self._audio_runners.pop(name, None)

        thread = threading.Thread(target=worker, daemon=True, name=f"audio-{name}")
        self._audio_runners[name] = (thread, stop_event)
        thread.start()
        self.viewmodel.log_event("INFO", f"audio started: {os.path.basename(path)}", name)

    def _stop_audio(self, name: str) -> None:
        runner = self._audio_runners.pop(name, None)
        if runner is None:
            return
        thread, stop_event = runner
        stop_event.set()
        thread.join(timeout=0.05)

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------
    def clear_all(self) -> None:
        self._stop_all_sources()
        self.files = {name: {} for name in self.viewmodel.approaches}
        self.frames.clear()
        self.viewmodel.clear_all()
        self._render_now()

    def clear_fault(self) -> None:
        self.viewmodel.clear_fault()
        self._render_now()

    def simulate_fault(self) -> None:
        self.viewmodel.trigger_fault()
        self._render_now()

    def _stop_all_sources(self) -> None:
        for name in list(self.viewmodel.approaches):
            self._stop_video(name)
            self._stop_audio(name)

    # ------------------------------------------------------------------
    # Main-thread refresh loop
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if self._closing:
            return
        try:
            for name, pipeline in list(self.pipelines.items()):
                stats = pipeline.stats
                self.viewmodel.update_video_stats(
                    name,
                    fps=stats.fps,
                    latency_ms=stats.avg_latency_ms,
                    frames_dropped=stats.frames_dropped,
                    last_error=stats.last_error,
                )

            now = self._clock()
            if now - self._last_health_sample >= self.config.monitoring.metrics_interval_s:
                self._last_health_sample = now
                sample = self._resources.sample()
                self.viewmodel.update_health(
                    cpu_percent=sample.cpu_percent,
                    system_ram_percent=sample.system_ram_percent,
                    process_ram_mb=sample.process_ram_mb,
                )

            snapshot = self.viewmodel.tick()
            self._render(snapshot)
        except Exception as exc:  # pragma: no cover - defensive UI loop guard
            try:
                self.viewmodel.log_event("ERROR", f"dashboard refresh error: {exc}")
            except Exception:
                pass
        finally:
            if not self._closing:
                self.root.after(REFRESH_INTERVAL_MS, self.refresh)

    def _render_now(self) -> None:
        try:
            self._render(self.viewmodel.snapshot())
        except Exception:
            pass

    def _render(self, snapshot: DashboardSnapshot) -> None:
        level = snapshot.status_level
        colour = {"ERROR": "#e74c3c", "WARNING": "#f39c12"}.get(level, TEXT_COLOR)
        state = "RUNNING" if snapshot.running else "STOPPED"
        self.status_label.configure(
            text=f"{state} | {snapshot.status_message}", fg=colour
        )
        self.health_panel.render(snapshot)

        for name, card in self._cards.items():
            view = snapshot.approach(name)
            stored = self.frames.get(name)
            image = None
            if stored is not None:
                rgb, bbox = stored
                pil = Image.fromarray(rgb)
                if bbox is not None:
                    ImageDraw.Draw(pil).rectangle(bbox, outline="#2ecc71", width=2)
                image = ImageTk.PhotoImage(pil)
            card.render(view, image)

        self.event_log.render(snapshot.events)

    def on_close(self) -> None:
        self._closing = True
        self._stop_all_sources()
        self.viewmodel.stop()
        self.root.destroy()


def main(config_path: Optional[str] = None) -> int:
    """Launch the dashboard. Kept as the body of `python new.py`."""
    config = load_config(config_path) if config_path else load_config()
    configure_logging(
        config.logging.level,
        text_file=config.logging.text_file,
        json_file=config.logging.json_file,
    )

    root = tk.Tk()
    try:
        DashboardApp(root, config)
    except Exception as exc:
        messagebox.showerror("Ambulance dashboard", f"Could not start the dashboard:\n{exc}")
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
