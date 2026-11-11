from __future__ import annotations

from pathlib import Path

import pytest

import ambulance_system.hardware as hardware
from ambulance_system.hardware import (
    BaseHardwareController,
    HardwareController,
    HardwareError,
    HardwareNotConnectedError,
    MockController,
    SimulationController,
    create_controller,
)
from ambulance_system.traffic_signal.states import SignalColor

APPROACHES = ("Road 1", "Road 2", "Road 3", "Road 4")


def test_simulation_starts_disconnected_and_all_red():
    controller = SimulationController(APPROACHES)

    assert controller.is_connected is False
    assert controller.approaches == APPROACHES
    assert controller.snapshot() == {name: SignalColor.RED for name in APPROACHES}
    assert controller.history == ()


def test_connect_forces_all_red_and_is_idempotent():
    controller = SimulationController(APPROACHES)

    controller.connect()
    assert controller.is_connected is True
    assert all(colour is SignalColor.RED for colour in controller.snapshot().values())

    controller.connect()
    assert controller.is_connected is True


def test_apply_requires_a_connection():
    controller = SimulationController(APPROACHES)

    with pytest.raises(HardwareNotConnectedError):
        controller.apply("Road 1", SignalColor.GREEN)


def test_apply_updates_snapshot_and_records_history():
    ticks = iter((10.0, 20.0))
    controller = SimulationController(APPROACHES, clock=lambda: next(ticks))
    controller.connect()

    controller.apply("Road 1", SignalColor.GREEN)
    controller.apply("Road 3", SignalColor.YELLOW)

    snapshot = controller.snapshot()
    assert snapshot["Road 1"] is SignalColor.GREEN
    assert snapshot["Road 3"] is SignalColor.YELLOW
    assert [command.to_dict() for command in controller.history] == [
        {"approach": "Road 1", "colour": "green", "timestamp": 10.0},
        {"approach": "Road 3", "colour": "yellow", "timestamp": 20.0},
    ]


def test_apply_rejects_unknown_approach_and_wrong_colour():
    controller = SimulationController(APPROACHES)
    controller.connect()

    with pytest.raises(ValueError, match="unknown approach"):
        controller.apply("Road 9", SignalColor.GREEN)
    with pytest.raises(ValueError, match="SignalColor"):
        controller.apply("Road 1", "green")


def test_snapshot_is_a_copy():
    controller = SimulationController(APPROACHES)
    controller.connect()

    snapshot = controller.snapshot()
    snapshot["Road 1"] = SignalColor.GREEN

    assert controller.snapshot()["Road 1"] is SignalColor.RED


def test_disconnect_forces_all_red_and_blocks_further_writes():
    controller = SimulationController(APPROACHES)
    controller.connect()
    controller.apply("Road 1", SignalColor.GREEN)

    controller.disconnect()

    assert controller.is_connected is False
    assert all(colour is SignalColor.RED for colour in controller.snapshot().values())
    with pytest.raises(HardwareNotConnectedError):
        controller.apply("Road 1", SignalColor.GREEN)

    history = controller.history
    controller.disconnect()
    assert controller.history == history


def test_apply_all_red_while_connected_records_commands():
    controller = SimulationController(APPROACHES)
    controller.connect()
    controller.apply("Road 1", SignalColor.GREEN)
    controller.clear_history()

    controller.apply_all_red()

    assert all(colour is SignalColor.RED for colour in controller.snapshot().values())
    assert [command.approach for command in controller.history] == list(APPROACHES)


def test_apply_all_red_while_disconnected_only_sets_safe_state():
    controller = SimulationController(APPROACHES)

    controller.apply_all_red()

    assert all(colour is SignalColor.RED for colour in controller.snapshot().values())
    assert controller.history == ()


def test_context_manager_connects_and_disconnects():
    with SimulationController(APPROACHES) as controller:
        assert controller.is_connected is True
        controller.apply("Road 2", SignalColor.GREEN)

    assert controller.is_connected is False
    assert all(colour is SignalColor.RED for colour in controller.snapshot().values())


def test_base_controller_has_no_output_transport():
    controller = BaseHardwareController(APPROACHES)
    controller.connect()

    with pytest.raises(NotImplementedError):
        controller.apply("Road 1", SignalColor.GREEN)


def test_mock_records_writes_and_call_counts():
    controller = MockController(APPROACHES)
    controller.connect()
    controller.apply("Road 2", SignalColor.GREEN)
    controller.disconnect()

    assert controller.writes == [("Road 2", SignalColor.GREEN)]
    assert controller.connect_calls == 1
    assert controller.disconnect_calls == 1

    controller.reset()
    assert controller.writes == []
    assert controller.connect_calls == 0


def test_mock_connect_failure_leaves_it_disconnected_and_safe():
    controller = MockController(APPROACHES, fail_on_connect=True)

    with pytest.raises(HardwareError, match="connect"):
        controller.connect()

    assert controller.is_connected is False
    assert all(colour is SignalColor.RED for colour in controller.snapshot().values())


def test_mock_write_failure_keeps_the_previous_state():
    controller = MockController(APPROACHES, fail_on_apply=True)
    controller.connect()

    with pytest.raises(HardwareError, match="apply"):
        controller.apply("Road 1", SignalColor.GREEN)

    assert controller.snapshot()["Road 1"] is SignalColor.RED
    assert controller.writes == []


def test_invalid_approach_configuration_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SimulationController(())
    with pytest.raises(ValueError, match="non-empty"):
        SimulationController(("Road 1", "  "))
    with pytest.raises(ValueError, match="unique"):
        SimulationController(("Road 1", "Road 1"))


def test_controllers_are_structurally_hardware_controllers():
    assert isinstance(SimulationController(APPROACHES), HardwareController)
    assert isinstance(MockController(APPROACHES), HardwareController)


def test_factory_builds_simulation_and_mock_and_rejects_unknown_kinds():
    assert isinstance(create_controller("simulation"), SimulationController)
    assert isinstance(create_controller("mock"), MockController)

    with pytest.raises(ValueError, match="unsupported hardware controller"):
        create_controller("gpio")


def test_hardware_package_imports_no_device_transport_library():
    package = Path(hardware.__file__).parent
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )

    for forbidden in (
        "import serial",
        "import RPi",
        "import gpiozero",
        "import smbus",
        "import spidev",
        "import socket",
    ):
        assert forbidden not in source
