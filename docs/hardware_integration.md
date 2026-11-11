# Hardware Integration: Simulation and Mock Only

**Status:** Stage 14 implemented. Software-only abstraction; **no physical signal
hardware has been connected, opened or tested.**
**Last updated:** 2026-11-12
**Scope:** `src/ambulance_system/hardware/` and `tests/test_hardware.py`

This document describes the hardware boundary added in Stage 14. It exists so a
future controller can be added behind a small interface without changing the
fusion or traffic-signal code. It does **not** claim that any real traffic light,
relay, microcontroller or GPIO pin has been exercised.

---

## 0. What is included

| Component | Purpose |
|---|---|
| `hardware/controller.py` | `HardwareController` protocol, `BaseHardwareController`, `HardwareCommand`, errors and safe defaults |
| `hardware/simulation.py` | `SimulationController` — the default, in-memory controller with a timestamped history |
| `hardware/mock.py` | `MockController` — deterministic test double that can simulate connect/write failures |
| `tests/test_hardware.py` | 18 tests for connection lifecycle, safe defaults and failure paths |
| `create_controller(kind, ...)` | Factory accepting only `"simulation"` and `"mock"` |

The package imports **no** GPIO, relay, serial or network library. A test scans
the package source and fails if a forbidden import appears.

---

## 1. Interface

```python
class HardwareController(Protocol):
    is_connected: bool
    approaches: tuple[str, ...]
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def apply(self, approach: str, colour: SignalColor) -> None: ...
    def apply_all_red(self) -> None: ...
    def snapshot(self) -> Mapping[str, SignalColor]: ...
```

The interface deliberately does **not** validate phase transitions. That remains
the responsibility of `traffic_signal.states`, which already enforces the legal
RED/YELLOW/GREEN matrix. The hardware layer only mirrors accepted commands and
keeps a safe state.

---

## 2. Safety invariants

These are enforced by `BaseHardwareController` and covered by tests:

1. A controller starts **disconnected** with every output **red**.
2. `connect()` forces every output red before accepting commands.
3. `disconnect()` forces every output red **before** closing the transport.
4. `apply()` raises `HardwareNotConnectedError` unless the controller is connected.
5. An unknown approach or a non-`SignalColor` value raises `ValueError`.
6. A failing transport write raises and does **not** update the recorded state.
7. `apply_all_red()` always leaves every output red; while disconnected it only
   updates the in-memory safe state and writes nothing.
8. `snapshot()` returns a copy, so callers cannot mutate internal state.

The default `signal.fail_safe_state: ALL_RED` in `config/default.yaml` matches
this behaviour.

---

## 3. Usage

```python
from ambulance_system.hardware import create_controller
from ambulance_system.traffic_signal.states import SignalColor

controller = create_controller("simulation", ("Road 1", "Road 2", "Road 3", "Road 4"))
controller.connect()                      # all red
controller.apply("Road 1", SignalColor.GREEN)
controller.apply("Road 1", SignalColor.RED)
controller.disconnect()                   # forced all red

for command in controller.history:
    print(command.to_dict())
```

Using it as a context manager disconnects automatically:

```python
with create_controller("mock") as controller:
    controller.apply("Road 2", SignalColor.YELLOW)
# left disconnected and all-red
```

`SimulationController` accepts an injectable `clock` (default `time.monotonic`),
so timestamps are deterministic in tests. `MockController` exposes `writes`,
`connect_calls` and `disconnect_calls`, and accepts `fail_on_connect` /
`fail_on_apply` to exercise failure paths.

---

## 4. Logging

The base controller logs `hardware.connect`, `hardware.disconnect` and
`hardware.apply` through `ambulance_system.logging_utils`. The apply event carries
the approach name. No pin numbers, device paths or credentials are logged because
none exist in this layer.

---

## 5. What is explicitly not included

- No GPIO, relay, serial (RS-485), Modbus, I2C or SPI access.
- No microcontroller firmware, network protocol or cloud endpoint.
- No claim of electrical isolation, timing accuracy or fail-safe hardware.
- No wiring into the dashboard runtime. The dashboard remains a software
  simulation and does not drive any physical output.
- No physical hardware test. All 18 tests use in-memory controllers.

---

## 6. Requirements before any physical prototype

A future physical adapter is allowed only under explicit approval and must:

1. Implement `HardwareController` without weakening the invariants in section 2.
2. Start disconnected and all-red; never energise a green aspect on import.
3. Keep a **manual emergency shutdown** outside this software.
4. Use isolated, low-voltage lab equipment under direct human supervision.
5. Be tested on a bench with lamps or LEDs before any road-side use.
6. Document the exact device, wiring, power and failure behaviour, and state
   clearly that the road-side system remains a simulation until then.

Do not describe the project as controlling real traffic signals on the strength
of this abstraction. It does not.

---

## 7. Tests

```powershell
.\venv\Scripts\python.exe -m pytest -q tests\test_hardware.py
```

Covered: initial safe state, idempotent connect, connection requirement,
colour/approach validation, snapshot isolation, disconnect forcing red,
`apply_all_red` in both connection states, context-manager cleanup, base-class
transport rejection, mock call recording, connect and write failures, approach
configuration validation, protocol conformance, factory rejection of unknown
kinds, and the forbidden-import source scan.
