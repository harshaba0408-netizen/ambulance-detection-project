"""Default software-only signal controller.

`SimulationController` is the controller the project uses by default. It keeps
the last colour per approach in memory and records every accepted command with a
timestamp. It never imports a GPIO, serial or network library and never touches a
real signal head.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

from ambulance_system.hardware.controller import BaseHardwareController, HardwareCommand
from ambulance_system.traffic_signal.states import SignalColor


class SimulationController(BaseHardwareController):
    """An in-memory controller with a timestamped command history.

    `clock` is injectable so tests can pin timestamps without sleeping. The
    default is `time.monotonic`.
    """

    def __init__(
        self,
        approaches: Optional[Iterable[str]] = None,
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(approaches)
        self._clock = clock or time.monotonic
        self._history: list[HardwareCommand] = []

    @property
    def history(self) -> tuple[HardwareCommand, ...]:
        """Every colour accepted so far, oldest first."""
        return tuple(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def _write(self, approach: str, colour: SignalColor) -> None:
        self._history.append(
            HardwareCommand(
                approach=approach,
                colour=colour,
                timestamp=float(self._clock()),
            )
        )
