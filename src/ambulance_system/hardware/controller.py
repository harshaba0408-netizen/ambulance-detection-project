"""Hardware abstraction for traffic-signal outputs: simulation and mock only.

Nothing in this package imports or opens a GPIO pin, relay, serial port or
network device. The default `SimulationController` keeps every output in memory,
so the rest of the system can run without traffic hardware. A future physical
prototype must implement `HardwareController` behind this interface, start in an
all-red state, and keep a manual emergency shutdown outside this software.

Safety invariants enforced here:

* a controller starts disconnected with every output red;
* `connect()` and `disconnect()` both leave every output red;
* a colour can only be applied while connected;
* a failed transport write does not change the recorded output.

The interface is intentionally small. It does not validate phase transitions;
that remains the job of `traffic_signal`. It deliberately does not claim that any
physical signal was tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Protocol, runtime_checkable

from ambulance_system.logging_utils import get_logger
from ambulance_system.traffic_signal.states import SignalColor

#: The only colour the simulator permits when no command has been applied.
SAFE_COLOUR = SignalColor.RED

#: Default approach names, matching `config/default.yaml`.
DEFAULT_APPROACHES = ("Road 1", "Road 2", "Road 3", "Road 4")


class HardwareError(RuntimeError):
    """Base error for the hardware abstraction."""


class HardwareNotConnectedError(HardwareError):
    """Raised when a colour is applied before the controller is connected."""


@dataclass(frozen=True)
class HardwareCommand:
    """One colour command applied to one approach."""

    approach: str
    colour: SignalColor
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": self.approach,
            "colour": self.colour.value,
            "timestamp": float(self.timestamp),
        }


def _validated_approaches(approaches: Optional[Iterable[str]]) -> tuple[str, ...]:
    names = tuple(DEFAULT_APPROACHES if approaches is None else approaches)
    if not names:
        raise ValueError("at least one approach is required")
    cleaned: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("approach names must be non-empty strings")
        cleaned.append(name.strip())
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("approach names must be unique")
    return tuple(cleaned)


@runtime_checkable
class HardwareController(Protocol):
    """Minimal interface behind which a future physical controller could sit."""

    @property
    def is_connected(self) -> bool:
        """True only after a successful `connect()`."""

    @property
    def approaches(self) -> tuple[str, ...]:
        """The approach names this controller accepts."""

    def connect(self) -> None:
        """Open the transport and force the all-red safe state."""

    def disconnect(self) -> None:
        """Force the all-red safe state and close the transport."""

    def apply(self, approach: str, colour: SignalColor) -> None:
        """Apply one colour while connected."""

    def apply_all_red(self) -> None:
        """Force every known approach to red."""

    def snapshot(self) -> Mapping[str, SignalColor]:
        """Return a copy of the last applied colour per approach."""


class BaseHardwareController:
    """Shared safe-default behaviour for simulation and mock controllers."""

    def __init__(self, approaches: Optional[Iterable[str]] = None) -> None:
        self._approaches = _validated_approaches(approaches)
        self._outputs: dict[str, SignalColor] = {
            name: SAFE_COLOUR for name in self._approaches
        }
        self._connected = False
        self._log = get_logger("hardware")

    @property
    def approaches(self) -> tuple[str, ...]:
        return self._approaches

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return
        self._on_connect()
        self._outputs = {name: SAFE_COLOUR for name in self._approaches}
        self._connected = True
        self._log.info(
            "hardware controller connected in all-red safe state",
            extra={"event": "hardware.connect"},
        )

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._outputs = {name: SAFE_COLOUR for name in self._approaches}
        self._on_disconnect()
        self._connected = False
        self._log.info(
            "hardware controller disconnected; outputs forced all-red",
            extra={"event": "hardware.disconnect"},
        )

    def apply(self, approach: str, colour: SignalColor) -> None:
        if not self._connected:
            raise HardwareNotConnectedError(
                "connect() must succeed before applying a signal colour"
            )
        self._validate_approach(approach)
        if not isinstance(colour, SignalColor):
            raise ValueError("colour must be a SignalColor")
        # Write first: a failing transport must not update the recorded state.
        self._write(approach, colour)
        self._outputs[approach] = colour
        self._log.info(
            "hardware colour applied",
            extra={"event": "hardware.apply", "approach": approach},
        )

    def apply_all_red(self) -> None:
        """Force every output red, whether connected or not."""
        for approach in self._approaches:
            if self._connected:
                self._write(approach, SAFE_COLOUR)
            self._outputs[approach] = SAFE_COLOUR

    def snapshot(self) -> Mapping[str, SignalColor]:
        return dict(self._outputs)

    def _validate_approach(self, approach: str) -> None:
        if approach not in self._outputs:
            raise ValueError(f"unknown approach: {approach!r}")

    def _on_connect(self) -> None:
        """Hook for subclasses; raise to abort the connection."""

    def _on_disconnect(self) -> None:
        """Hook for subclasses."""

    def _write(self, approach: str, colour: SignalColor) -> None:
        """Hook for the subclass transport; raise to reject the command."""
        raise NotImplementedError("base controller has no output transport")

    def __enter__(self) -> "BaseHardwareController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.disconnect()
