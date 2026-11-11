"""Render-only tests for Stage 8 dashboard widgets.

These tests create a withdrawn Tk root but never open a window, start a worker
or run inference. They are skipped only when Tk cannot create a display.
"""

from __future__ import annotations

import tkinter as tk

import pytest

from ambulance_system.dashboard.viewmodel import (
    ApproachView,
    DashboardSnapshot,
    EventLogEntry,
    HealthView,
    ModalityView,
)
from ambulance_system.dashboard.widgets import (
    ApproachCard,
    EventLogPanel,
    HealthPanel,
    SignalCanvas,
    _duration,
    _megabytes,
    _modality_text,
    _percent,
)


@pytest.fixture
def root(tk_window):
    return tk_window


def approach_view(**overrides) -> ApproachView:
    defaults = dict(
        name="Road 1",
        policy="AUDIO",
        audio=ModalityView(active=True, confidence=0.82, last_seen_at=1.0),
        video=ModalityView(active=False, confidence=None, last_seen_at=2.0),
        fused_active=True,
        fused_confidence=0.82,
        fused_reason="fresh audio evidence",
        phase="emergency_priority",
        colour="green",
        priority_started_at=1.0,
        last_change_at=2.0,
        signal_reason="priority confirmed",
        video_fps=12.5,
        video_latency_ms=340.0,
        frames_dropped=7,
        video_error=None,
        bbox=(1.0, 2.0, 3.0, 4.0),
        frame_index=9,
    )
    defaults.update(overrides)
    return ApproachView(**defaults)


def snapshot(**overrides) -> DashboardSnapshot:
    defaults = dict(
        timestamp=10.0,
        running=True,
        approaches=(approach_view(),),
        active_approach="Road 1",
        fault=False,
        health=HealthView(
            cpu_percent=41.0,
            system_ram_percent=None,
            process_ram_mb=512.0,
            uptime_s=65.0,
            errors=2,
        ),
        events=(),
        status_message="running",
        status_level="INFO",
    )
    defaults.update(overrides)
    return DashboardSnapshot(**defaults)


def test_signal_canvas_lights_one_lamp(root):
    canvas = SignalCanvas(root)

    for colour in ("red", "yellow", "green"):
        canvas.render(colour)
        fills = [
            canvas.itemcget(item, "fill")
            for item in canvas.find_all()
            if canvas.type(item) == "oval"
        ]
        assert fills.count(colour) == 1
        assert fills.count("grey") == 2

    canvas.render("purple")
    fills = [
        canvas.itemcget(item, "fill")
        for item in canvas.find_all()
        if canvas.type(item) == "oval"
    ]
    assert fills.count("red") == 1


def test_approach_card_renders_evidence_and_policy(root):
    changes: list[str] = []
    card = ApproachCard(
        root,
        "Road 1",
        video_size=(64, 36),
        on_select_video=lambda: None,
        on_select_audio=lambda: None,
        on_start=lambda: None,
        on_stop=lambda: None,
        on_policy_change=changes.append,
    )
    view = approach_view()

    card.render(view)
    assert card.policy_var.get() == "AUDIO"
    assert "Audio: ON (0.82)" in card.audio_label.cget("text")
    assert "Video: OFF" in card.video_label_text.cget("text")
    assert "Fused: ACTIVE 0.82" in card.fused_label.cget("text")
    assert "GREEN / EMERGENCY_PRIORITY" in card.phase_label.cget("text")
    assert "FPS: 12.5" in card.metrics_label.cget("text")
    assert "latency: 340 ms" in card.metrics_label.cget("text")
    assert "dropped: 7" in card.metrics_label.cget("text")

    inert = approach_view(
        policy="VIDEO",
        fused_active=False,
        fused_confidence=None,
        fused_reason="waiting for evaluation",
        colour="red",
        phase="normal",
    )
    card.render(inert)
    assert card.policy_var.get() == "VIDEO"
    assert "Fused: idle (waiting for evaluation)" in card.fused_label.cget("text")
    assert "RED / NORMAL" in card.phase_label.cget("text")

    card._policy_changed("ADAPTIVE")
    assert changes == ["ADAPTIVE"]


def test_event_log_appends_once_and_trims(root):
    panel = EventLogPanel(root, max_lines=2)
    first = EventLogEntry(1, 0.0, "INFO", "first event", "Road 1")
    second = EventLogEntry(2, 0.0, "ERROR", "second event")
    third = EventLogEntry(3, 0.0, "WARNING", "third event", "Road 2")

    panel.render((first, second))
    text = panel.text.get("1.0", "end-1c")
    assert "[Road 1] first event" in text
    assert "ERROR second event" in text

    panel.render((first, second))
    assert panel.text.get("1.0", "end-1c") == text

    panel.render((first, second, third))
    lines = panel.text.get("1.0", "end-1c").splitlines()
    assert len(lines) <= 2
    assert "third event" in panel.text.get("1.0", "end-1c")


def test_health_panel_renders_measured_values(root):
    panel = HealthPanel(root)
    panel.render(snapshot())

    assert panel._labels["active"].cget("text") == "Road 1"
    assert panel._labels["fault"].cget("text") == "no"
    assert panel._labels["cpu"].cget("text") == "41%"
    assert panel._labels["ram"].cget("text") == "--"
    assert panel._labels["process"].cget("text") == "512 MB"
    assert panel._labels["uptime"].cget("text") == "1:05"
    assert panel._labels["errors"].cget("text") == "2"

    faulted = snapshot(fault=True, active_approach=None)
    panel.render(faulted)
    assert panel._labels["active"].cget("text") == "none"
    assert panel._labels["fault"].cget("text") == "YES"


def test_small_formatting_helpers_handle_missing_values():
    assert _modality_text("Audio", False, None) == "Audio: OFF"
    assert _modality_text("Audio", True, 0.5) == "Audio: ON (0.50)"
    assert _percent(None) == "--"
    assert _percent(42.4) == "42%"
    assert _megabytes(None) == "--"
    assert _megabytes(512.6) == "513 MB"
    assert _duration(59) == "0:59"
    assert _duration(3661) == "1:01:01"
