# Signal State Machine

**Stage 3 deliverable — software simulation only.**
This controller is not certified for real traffic control and must never be
connected to public road infrastructure.

The implementation lives in `src/ambulance_system/traffic_signal/`:

| File | Responsibility |
|---|---|
| `states.py` | Enums, transition matrix, `SignalCommand`, `ApproachObservation`, `ApproachSnapshot` |
| `machine.py` | Per-approach timing, hysteresis, clearance sequence |
| `arbiter.py` | Cross-approach arbitration, all-red safety interval, FAULT handling |

## 1. Two-level state model

The driver-visible colour and the controller's decision phase are separate:

```
SignalColor   : RED | YELLOW | GREEN
PriorityPhase : NORMAL | EMERGENCY_PRIORITY | CLEARING | FAULT
```

| Phase | Allowed colours | Meaning |
|---|---|---|
| `NORMAL` | RED | No priority; the simulator idles at all-red |
| `EMERGENCY_PRIORITY` | GREEN | Confirmed emergency for this approach |
| `CLEARING` | YELLOW, then RED | Priority released; yellow and all-red are running |
| `FAULT` | RED | Safety fault; all approaches forced red |

An approach is never GREEN in any phase other than `EMERGENCY_PRIORITY`, and
never YELLOW outside `CLEARING`. `validate_transition()` rejects every other
combination, including GREEN → RED without a YELLOW interval.

## 2. Transition table

| From | To | Colour | Trigger |
|---|---|---|---|
| `NORMAL` | `EMERGENCY_PRIORITY` | RED → GREEN | Request confirmed and all-red interval satisfied |
| `NORMAL` | `FAULT` | RED → RED | `trigger_fault()` |
| `EMERGENCY_PRIORITY` | `CLEARING` | GREEN → YELLOW | Release condition, grace period, max duration, or minimum green satisfied |
| `EMERGENCY_PRIORITY` | `FAULT` | GREEN → RED | `trigger_fault()` |
| `CLEARING` | `CLEARING` | YELLOW → RED | `yellow_s` elapsed |
| `CLEARING` | `NORMAL` | RED → RED | `yellow_s + all_red_s` elapsed |
| `CLEARING` | `FAULT` | YELLOW/RED → RED | `trigger_fault()` |
| `FAULT` | `NORMAL` | RED → RED | `clear_fault()` only |

There is no direct `EMERGENCY_PRIORITY → NORMAL`. `emergency_reset()` and
`clear_all()` are explicit safety actions that force all-red; they do not pass
through the normal transition table because every state is safe to exit to red.

## 3. Timing parameters

All values come from `config/default.yaml` under `signal:`.

| Parameter | Default | Effect |
|---|---:|---|
| `activation_threshold` | 0.65 | Confidence needed to start confirmation |
| `release_threshold` | 0.50 | Confidence below which release may start |
| `min_confirmation_s` | 1.0 | Continuous evidence required before a request is ready |
| `release_grace_s` | 2.0 | How long evidence may be absent before release |
| `min_green_s` | 5.0 | Minimum green before clearing may start |
| `max_priority_s` | 30.0 | Hard cap on one priority event |
| `yellow_s` | 3.0 | Yellow interval |
| `all_red_s` | 2.0 | All-red interval between greens |
| `fail_safe_state` | `ALL_RED` | Only state permitted in the simulator |

Hysteresis: confidence between `release_threshold` and
`activation_threshold` neither starts a request nor releases an active one.
A single missed frame cannot release priority; absence must persist for
`release_grace_s` and `min_green_s` must already be satisfied.

## 4. Decision cycle

`SignalController.step(observations, now)` performs, in order:

1. Reject unknown approaches, mismatched keys and backwards time.
2. Advance any active clearance (`tick`).
3. Feed each approach its observation. A missing approach is treated as
   **no evidence**, so a stalled producer cannot hold priority forever.
4. Start clearing any active approach whose release is due.
5. Maintain a pending set containing only confirmed requests from approaches
   in `NORMAL`.
6. Grant at most one priority if the junction is all-red and the all-red
   interval has elapsed.
7. Assert safety invariants.

Only actual changes produce `SignalCommand` objects. Each command carries a
unique `command_id`, timestamp (epoch + ISO-8601 UTC), approach, colour, phase
and reason, and is written to the structured log.

## 5. Conflict resolution

* The earliest confirmed request wins.
* Ties are broken by the configured approach order, so results are
  deterministic and reproducible.
* An active priority is **not preempted** by a later request. The later
  request waits in the pending set.
* Only one approach may be GREEN. A second green is impossible because a grant
  requires every other approach to be `NORMAL` / RED.
* A new green waits for the previous approach to finish YELLOW **and** for the
  full `all_red_s` interval.

## 6. Safety actions

| Method | Result | Recovery |
|---|---|---|
| `clear_all(now)` | All approaches NORMAL / RED | — |
| `emergency_reset(reason, now)` | All approaches NORMAL / RED, timers cleared | — |
| `trigger_fault(reason, now)` | All approaches FAULT / RED, pending cleared | required |
| `clear_fault(now)` | All-red NORMAL | explicit only |

While any approach is in `FAULT`, `step()` ignores observations and returns no
commands. This prevents uncertain model output from producing a signal change
during a fault.

`SafetyViolationError` is raised if an invariant would be broken (multiple
greens, green with yellow, or a non-red colour during FAULT). It indicates a
controller bug, not normal operation.

## 7. Usage

```python
from dataclasses import replace
from ambulance_system.config import SignalConfig, load_config
from ambulance_system.clock import FakeClock
from ambulance_system.traffic_signal import (
    ApproachObservation, SignalController, SignalColor,
)

config = load_config()
controller = SignalController(
    config.signal,
    config.system.approach_names,
    clock=FakeClock(start=0.0),   # RealClock in production
)

command = controller.step(
    {"Road 1": ApproachObservation("Road 1", requested=True, confidence=0.92)},
    now=1.0,
)
print(controller.states["Road 1"].colour)   # green once confirmed and safe
```

## 8. What Stage 3 does not do

* It does not read models, audio or video. Stage 7's fusion layer will produce
  `ApproachObservation` values from real evidence.
* It is not connected to `new.py` yet. The existing dashboard keeps its current
  behaviour until Stage 8 wires the controller in.
* It does not implement per-approach hardware output; Stage 14 provides the
  interface and mock controller.
* It does not claim real-world traffic safety. The invariants are simulator
  properties, verified by tests.

## 9. Verification

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_signal_machine.py tests\test_signal_arbiter.py -q
```

45 tests cover the transition matrix, confirmation, hysteresis, grace period,
minimum green, maximum duration, clearance sequencing, conflict resolution,
handover, stale evidence, manual reset and fault recovery. All timing tests use
`FakeClock`; no test sleeps.
