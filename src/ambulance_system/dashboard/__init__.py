"""Dashboard package (Stage 8).

The package root deliberately exports only headless view-model types. Importing
`ambulance_system.dashboard` must not import Tkinter; import
`ambulance_system.dashboard.app` or `.widgets` explicitly when a display is
available.
"""

from ambulance_system.dashboard.viewmodel import (
    ApproachView,
    DashboardSnapshot,
    DashboardViewModel,
    EventLogEntry,
    HealthView,
    ModalityView,
)

__all__ = [
    "ApproachView",
    "DashboardSnapshot",
    "DashboardViewModel",
    "EventLogEntry",
    "HealthView",
    "ModalityView",
]
