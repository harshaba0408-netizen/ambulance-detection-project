"""Modular core for the AI Ambulance Detection & Traffic Signal Priority prototype.

Stage 2 of the roadmap: configuration, event schema, clock abstraction and
structured logging. Nothing here imports Tkinter, Torch or a capture device,
so every module is safe to import from tests and headless tools.
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
