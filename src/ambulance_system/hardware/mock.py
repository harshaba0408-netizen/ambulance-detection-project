"""Deterministic in-memory mock for tests.

`MockController` records calls and can be told to fail on connect or write, so
failure paths can be exercised without a device. It is a test double, not a
simulation of signal timing.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ambulance_system.hardware.controller import BaseHardwareController, HardwareError
from ambulance_system.traffic_signal.states import SignalColor


class MockController(BaseHardwareController):
    """A programmable test double with no output transport."""

    def __init__(
        self,
        approaches: Optional[Iterable[str]] = None,
        *,
        fail_on_connect: bool = False,
        fail_on_apply: bool = False,
    ) -> None:
        super().__init__(approaches)
        self.fail_on_connect = bool(fail_on_connect)
        self.fail_on_apply = bool(fail_on_apply)
        self.writes: list[tuple[str, SignalColor]] = []
        self.connect_calls = 0
        self.disconnect_calls = 0

    def reset(self) -> None:
        """Clear recorded calls; connection state is left untouched."""
        self.writes.clear()
        self.connect_calls = 0
        self.disconnect_calls = 0

    def _on_connect(self) -> None:
        self.connect_calls += 1
        if self.fail_on_connect:
            raise HardwareError("mock connect failure")

    def _on_disconnect(self) -> None:
        self.disconnect_calls += 1

    def _write(self, approach: str, colour: SignalColor) -> None:
        if self.fail_on_apply:
            raise HardwareError("mock apply failure")
        self.writes.append((approach, colour))
