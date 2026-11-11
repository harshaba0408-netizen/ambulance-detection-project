"""Tkinter widgets that only render immutable dashboard snapshots.

No widget in this module starts a worker thread, opens a capture device or
performs inference. Every update arrives as an `ApproachView` /
`DashboardSnapshot` produced by `viewmodel.py`.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from ambulance_system.dashboard.viewmodel import (
    ApproachView,
    DashboardSnapshot,
    EventLogEntry,
)

BACKGROUND = "#233047"
CARD_BACKGROUND = "#2c3e50"
TEXT_COLOR = "white"
MUTED_COLOR = "#bdc3c7"
POLICY_CHOICES = ("AUDIO", "VIDEO", "BOTH", "ADAPTIVE")


class SignalCanvas(tk.Canvas):
    """Three-lamp signal head driven by `ApproachView.colour`."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(
            master,
            width=60,
            height=140,
            bg="black",
            highlightthickness=0,
            **kwargs,
        )
        self.create_rectangle(15, 10, 45, 130, fill="grey25")
        self._red = self.create_oval(18, 13, 42, 37, fill="red")
        self._yellow = self.create_oval(18, 53, 42, 77, fill="grey")
        self._green = self.create_oval(18, 93, 42, 117, fill="grey")

    def render(self, colour: str) -> None:
        lit = {"red": self._red, "yellow": self._yellow, "green": self._green}
        if colour not in lit:
            colour = "red"
        for name, item in lit.items():
            self.itemconfig(item, fill=name if name == colour else "grey")


class ApproachCard(tk.LabelFrame):
    """One approach: preview, controls, evidence text and signal head."""

    def __init__(
        self,
        master: tk.Misc,
        name: str,
        *,
        video_size: tuple[int, int],
        on_select_video: Callable[[], None],
        on_select_audio: Callable[[], None],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_policy_change: Callable[[str], None],
    ) -> None:
        super().__init__(
            master,
            text=name,
            bg=CARD_BACKGROUND,
            fg=TEXT_COLOR,
            font=("Helvetica", 13, "bold"),
            padx=8,
            pady=8,
        )
        self.name = name
        self._on_policy_change = on_policy_change
        self._image = None
        width, height = video_size

        video_frame = tk.Frame(self, width=width, height=height, bg="black")
        video_frame.grid_propagate(False)
        video_frame.grid(row=0, column=0, columnspan=3, pady=(0, 6))

        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        controls = tk.Frame(self, bg=CARD_BACKGROUND)
        controls.grid(row=1, column=0, columnspan=3, sticky="w")

        ttk.Button(controls, text="Select Video", command=on_select_video).grid(
            row=0, column=0, padx=3, pady=2
        )
        ttk.Button(controls, text="Select Audio", command=on_select_audio).grid(
            row=0, column=1, padx=3, pady=2
        )
        ttk.Button(controls, text="Start", command=on_start).grid(
            row=0, column=2, padx=3, pady=2
        )
        ttk.Button(controls, text="Stop", command=on_stop).grid(
            row=0, column=3, padx=3, pady=2
        )

        tk.Label(self, text="Policy:", bg=CARD_BACKGROUND, fg=TEXT_COLOR).grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )
        self.policy_var = tk.StringVar(value="AUDIO")
        ttk.OptionMenu(
            self,
            self.policy_var,
            "AUDIO",
            *POLICY_CHOICES,
            command=self._policy_changed,
        ).grid(row=2, column=1, sticky="w", pady=(4, 0))

        self.signal = SignalCanvas(self)
        self.signal.grid(row=0, column=3, rowspan=5, padx=(10, 0))

        self.audio_label = tk.Label(self, text="Audio: OFF", bg=CARD_BACKGROUND, fg=TEXT_COLOR)
        self.audio_label.grid(row=3, column=0, columnspan=3, sticky="w")
        self.video_label_text = tk.Label(
            self, text="Video: OFF", bg=CARD_BACKGROUND, fg=TEXT_COLOR
        )
        self.video_label_text.grid(row=4, column=0, columnspan=3, sticky="w")
        self.fused_label = tk.Label(
            self, text="Fused: idle", bg=CARD_BACKGROUND, fg=MUTED_COLOR
        )
        self.fused_label.grid(row=5, column=0, columnspan=3, sticky="w")
        self.phase_label = tk.Label(
            self, text="Signal: RED / NORMAL", bg=CARD_BACKGROUND, fg=TEXT_COLOR
        )
        self.phase_label.grid(row=6, column=0, columnspan=3, sticky="w")
        self.metrics_label = tk.Label(
            self, text="FPS: -- | latency: -- | dropped: 0",
            bg=CARD_BACKGROUND, fg=MUTED_COLOR,
        )
        self.metrics_label.grid(row=7, column=0, columnspan=3, sticky="w")

    def _policy_changed(self, value: str) -> None:
        self._on_policy_change(value)

    def render(self, view: ApproachView, image=None) -> None:
        if view.policy != self.policy_var.get():
            self.policy_var.set(view.policy)

        if image is not None:
            self._image = image
            self.video_label.configure(image=image)

        self.audio_label.configure(text=_modality_text("Audio", view.audio.active, view.audio.confidence))
        self.video_label_text.configure(
            text=_modality_text("Video", view.video.active, view.video.confidence)
        )

        if view.fused_active:
            confidence = "" if view.fused_confidence is None else f" {view.fused_confidence:.2f}"
            self.fused_label.configure(text=f"Fused: ACTIVE{confidence}", fg="#2ecc71")
        else:
            self.fused_label.configure(text=f"Fused: idle ({view.fused_reason})", fg=MUTED_COLOR)

        self.phase_label.configure(
            text=f"Signal: {view.colour.upper()} / {view.phase.upper()}",
            fg=TEXT_COLOR if view.colour == "red" else "#2ecc71",
        )
        self.metrics_label.configure(
            text=(
                f"FPS: {view.video_fps:.1f} | latency: {view.video_latency_ms:.0f} ms"
                f" | dropped: {view.frames_dropped}"
            )
        )
        self.signal.render(view.colour)


def _modality_text(label: str, active: bool, confidence: Optional[float]) -> str:
    state = "ON" if active else "OFF"
    if confidence is None:
        return f"{label}: {state}"
    return f"{label}: {state} ({confidence:.2f})"


class EventLogPanel(tk.LabelFrame):
    """Append-only user-facing event history."""

    def __init__(self, master: tk.Misc, *, max_lines: int = 300) -> None:
        super().__init__(master, text="Event History", bg=BACKGROUND, fg=TEXT_COLOR)
        self._max_lines = max_lines
        self._last_sequence = 0

        self.text = tk.Text(self, height=8, bg="#ecf0f1", state="disabled", wrap="none")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text.tag_configure("ERROR", foreground="#c0392b")
        self.text.tag_configure("WARNING", foreground="#d35400")
        self.text.tag_configure("INFO", foreground="#2c3e50")

    def render(self, events: tuple[EventLogEntry, ...]) -> None:
        new_entries = [entry for entry in events if entry.sequence > self._last_sequence]
        if not new_entries:
            return

        self.text.configure(state="normal")
        for entry in new_entries:
            stamp = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            prefix = f"[{entry.approach}] " if entry.approach else ""
            self.text.insert("end", f"{stamp} {entry.level} {prefix}{entry.message}\n", entry.level)
        self._last_sequence = new_entries[-1].sequence
        self._trim()
        self.text.see("end")
        self.text.configure(state="disabled")

    def _trim(self) -> None:
        lines = self.text.get("1.0", "end-1c").splitlines()
        if len(lines) > self._max_lines:
            remove = len(lines) - self._max_lines
            self.text.delete("1.0", f"{remove + 1}.0")


class HealthPanel(tk.LabelFrame):
    """Measured health and reliability summary."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="System Health", bg=BACKGROUND, fg=TEXT_COLOR)
        self._labels: dict[str, tk.Label] = {}
        for row, key in enumerate(
            ("active", "fault", "cpu", "ram", "process", "uptime", "errors")
        ):
            title = key.capitalize()
            tk.Label(self, text=f"{title}:", bg=BACKGROUND, fg=MUTED_COLOR, anchor="w").grid(
                row=row, column=0, sticky="w", padx=(2, 6)
            )
            value = tk.Label(self, text="--", bg=BACKGROUND, fg=TEXT_COLOR, anchor="w")
            value.grid(row=row, column=1, sticky="w")
            self._labels[key] = value

    def render(self, snapshot: DashboardSnapshot) -> None:
        health = snapshot.health
        self._labels["active"].configure(text=snapshot.active_approach or "none")
        self._labels["fault"].configure(
            text="YES" if snapshot.fault else "no",
            fg="#e74c3c" if snapshot.fault else TEXT_COLOR,
        )
        self._labels["cpu"].configure(text=_percent(health.cpu_percent))
        self._labels["ram"].configure(text=_percent(health.system_ram_percent))
        self._labels["process"].configure(text=_megabytes(health.process_ram_mb))
        self._labels["uptime"].configure(text=_duration(health.uptime_s))
        self._labels["errors"].configure(
            text=str(health.errors), fg="#e74c3c" if health.errors else TEXT_COLOR
        )


def _percent(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:.0f}%"


def _megabytes(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:.0f} MB"


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"
