"""Hardware package: simulation and mock controllers only.

No real device is opened. See `docs/hardware_integration.md`.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ambulance_system.hardware.controller import (
    DEFAULT_APPROACHES,
    SAFE_COLOUR,
    BaseHardwareController,
    HardwareCommand,
    HardwareController,
    HardwareError,
    HardwareNotConnectedError,
)
from ambulance_system.hardware.mock import MockController
from ambulance_system.hardware.simulation import SimulationController

_CONTROLLERS = {
    "simulation": SimulationController,
    "mock": MockController,
}


def create_controller(
    kind: str = "simulation",
    approaches: Optional[Iterable[str]] = None,
    **options: Any,
) -> BaseHardwareController:
    """Build a controller by name.

    `kind` is `"simulation"` (default) or `"mock"`. Unknown kinds are rejected so
    a future physical backend cannot be enabled by accident.
    """
    try:
        factory = _CONTROLLERS[kind]
    except KeyError:
        raise ValueError(
            f"unsupported hardware controller {kind!r}; expected one of "
            f"{sorted(_CONTROLLERS)}"
        ) from None
    return factory(approaches, **options)


__all__ = [
    "DEFAULT_APPROACHES",
    "SAFE_COLOUR",
    "BaseHardwareController",
    "HardwareCommand",
    "HardwareController",
    "HardwareError",
    "HardwareNotConnectedError",
    "MockController",
    "SimulationController",
    "create_controller",
]
