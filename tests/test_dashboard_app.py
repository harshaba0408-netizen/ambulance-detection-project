"""Non-UI tests for the dashboard application shell.

`FrameStore` is deliberately independent of Tkinter, so it is tested without a
display. The remaining `DashboardApp` tests use a withdrawn Tk root and injected
fake models; they are skipped only when no display is available.
"""

from __future__ import annotations

import tkinter as tk

import numpy as np
import pytest

from ambulance_system.config import load_config
from ambulance_system.dashboard.app import DashboardApp, FrameStore


# ---------------------------------------------------------------------------
# FrameStore
# ---------------------------------------------------------------------------
def test_frame_store_resizes_bgr_frame_and_scales_bbox():
    store = FrameStore((32, 18))
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # red in BGR

    store.put("Road 1", frame, bbox=(0, 0, 64, 36))
    stored = store.get("Road 1")

    assert stored is not None
    rgb, bbox = stored
    assert rgb.shape == (18, 32, 3)
    assert rgb[0, 0, 0] == 255  # BGR red became RGB red
    assert bbox == (0.0, 0.0, 31.0, 17.0)


def test_frame_store_handles_grey_and_rgba_frames():
    store = FrameStore((8, 8))
    grey = np.full((8, 8, 1), 128, dtype=np.uint8)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255

    store.put("Road 1", grey)
    grey_rgb, _ = store.get("Road 1")
    assert grey_rgb.shape == (8, 8, 3)

    store.put("Road 2", rgba)
    rgba_rgb, _ = store.get("Road 2")
    assert rgba_rgb.shape == (8, 8, 3)


def test_frame_store_ignores_invalid_frames_and_clears():
    store = FrameStore((8, 8))
    store.put("Road 1", np.zeros((8, 8), dtype=np.uint8))  # missing channel axis
    store.put("Road 1", np.zeros((0, 8, 3), dtype=np.uint8))

    assert store.get("Road 1") is None
    assert store.get("Road 9") is None

    store.put("Road 1", np.zeros((8, 8, 3), dtype=np.uint8))
    store.clear()
    assert store.get("Road 1") is None


# ---------------------------------------------------------------------------
# DashboardApp
# ---------------------------------------------------------------------------
@pytest.fixture
def root(tk_window):
    return tk_window


def _app(root: tk.Tk) -> DashboardApp:
    # Injected fakes bypass model loading. No worker is started by construction.
    return DashboardApp(
        root,
        load_config(),
        detector=object(),
        classifier=object(),
    )


def test_dashboard_app_builds_renders_actions_and_closes(root, tmp_path):
    app = _app(root)

    assert set(app._cards) == {"Road 1", "Road 2", "Road 3", "Road 4"}
    app.frames.put("Road 1", np.zeros((12, 12, 3), dtype=np.uint8), bbox=(1, 2, 10, 11))

    # No sources selected: Start warns, Stop is harmless.
    app.start_approach(0)
    app.stop_approach(0)
    assert any("no video or audio file selected" in entry.message for entry in app.viewmodel.snapshot().events)

    # Policy changes exercise the source-sync branches without opening files.
    for policy in ("VIDEO", "AUDIO", "BOTH", "ADAPTIVE"):
        app.change_policy(0, policy)
    assert app.viewmodel.policy_for("Road 1") == "ADAPTIVE"

    app.refresh()
    root.update_idletasks()

    app.simulate_fault()
    assert app.viewmodel.snapshot().fault is True
    app.clear_fault()
    assert app.viewmodel.snapshot().fault is False

    app._start_video("Road 1", str(tmp_path / "missing.mp4"))
    assert any("video file not found" in entry.message for entry in app.viewmodel.snapshot().events)

    app.clear_all()
    app.on_close()
    assert app._closing is True


def test_dashboard_app_close_stops_viewmodel(root):
    app = _app(root)
    app.on_close()
    assert app.viewmodel.snapshot().running is False
