"""Shared test fixtures.

Only the dashboard rendering tests need Tk. One hidden root is created per test
session so the suite does not repeatedly initialise Tcl/Tk; individual tests get
a fresh `Toplevel` that can be destroyed without taking down the session root.
"""

from __future__ import annotations

import tkinter as tk

import pytest


@pytest.fixture(scope="session")
def tk_session_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display is not available: {exc}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def tk_window(tk_session_root):
    window = tk.Toplevel(tk_session_root)
    window.withdraw()
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass
