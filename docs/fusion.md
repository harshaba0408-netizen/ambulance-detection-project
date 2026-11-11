# Multimodal Fusion Engine

Stage 7 adds one component that decides whether an approach currently has an
emergency, instead of the dashboard combining independent audio and video
booleans. It consumes the shared `DetectionEvent` schema produced by Stage 4
(video) and Stage 6 (audio).

**Scope:** the engine and its tests are complete, and Stage 8 wires it into the
dashboard through `DashboardViewModel`. The Tk application no longer combines
scattered audio/video booleans itself. Nothing here changes traffic-signal
behaviour on its own; the Stage 3 controller remains the only component allowed
to grant GREEN.

---

## 1. Data flow

```
DetectionEvent ──► submit() ──► debounce by (approach, source, type)
                                     │ accepted
                                     ▼
                              per-approach buffer
                                     │ evaluate(now)
                                     ▼
                   prune events older than stale_after_s
                                     ▼
                          policy (VIDEO/AUDIO/BOTH/ADAPTIVE)
                                     ▼
             FusionDecision per approach + changed decisions only
                                     ▼
                  ApproachObservation ──► SignalController.step()
```

The controller from Stage 3 remains the only component allowed to grant GREEN.
Fusion can mark several approaches active at once; the controller resolves the
conflict deterministically.

## 2. Policies

| Policy | Active when | Confidence | Notes |
|---|---|---|---|
| `VIDEO` | at least one fresh video event | maximum fresh video confidence | Audio ignored |
| `AUDIO` | at least one fresh audio event | maximum fresh audio confidence | Video ignored |
| `BOTH` | fresh video **and** fresh audio | `min(video, audio)` | Conjunction; see below |
| `ADAPTIVE` | fresh video **and** fresh audio inside the tighter adaptive window | `min(video, audio)` | Never looser than `BOTH` |

### Confidence rules

Audio and video confidences are **never averaged**. They measure different
things and are not calibrated against each other.

* Single-modality policies report the highest confidence among that modality's
  fresh events.
* `BOTH` and `ADAPTIVE` require both modalities, so they report the **minimum**
  of the two modality maxima. This is the conservative bound implied by a
  conjunction. For example, video 0.9 and audio 0.2 yields 0.2, not 0.55.

### ADAPTIVE rules

`ADAPTIVE` is not a looser `BOTH`. It is exactly `BOTH` plus an extra temporal
constraint:

1. Both modalities must have at least one event whose age is at most
   `fusion.adaptive_window_s`.
2. `adaptive_window_s` is validated to be positive and no larger than
   `stale_after_s`; the default is 1.0 s against a 3.0 s staleness window.
3. Confidence is the same conservative minimum as `BOTH`.
4. No confirmation requirement is reduced. Signal confirmation, activation
   threshold and hysteresis remain in the Stage 3 machine.

With the defaults, a video event aged 2 s and an audio event aged 0 s makes
`BOTH` active but `ADAPTIVE` inactive.

## 3. Staleness

An event is fresh when:

```
now - event.timestamp <= fusion.stale_after_s
```

The boundary is inclusive: an event exactly `stale_after_s` old is still fresh.
`evaluate(now)` prunes expired events and counts them in `stats.expired`.
Missing evidence therefore produces an inactive decision after the window,
instead of holding priority forever.

`evaluate` rejects a `now` earlier than the previous cycle, so pruning cannot be
undone by a backwards clock.

## 4. Duplicate suppression

The key is `(approach, source, detection_type)`. A second event with the same
key is suppressed while:

```
event.timestamp - last_accepted_timestamp < fusion.debounce_s
```

* Exactly `debounce_s` later is accepted; only strictly earlier is suppressed.
* Out-of-order events for the same key are suppressed.
* The same detection from a different source, or for a different approach, is a
  different key and is not suppressed.

Suppressed and accepted counts are available in `FusionEngine.stats`.

## 5. State changes and traceability

`evaluate()` always returns a `FusionDecision` for every configured approach.
`update.changed` contains only approaches whose `active` flag flipped in this
cycle.

Each decision carries:

* `approach`, `active`, `confidence` (or `None`),
* `reason` (for example `"fresh video evidence"`),
* `evidence_ids` — sorted IDs of the events that were used,
* `timestamp`.

An activation therefore names the evidence that caused it. A deactivation caused
by expiry has no fresh evidence and carries an empty tuple.

Conversions:

```python
decision.to_observation()      # ApproachObservation for SignalController
decision.to_priority_request() # PriorityRequest with the same evidence IDs
```

Inactive decisions use confidence `0.0` in the controller observation, because
`ApproachObservation` requires a number. The `PriorityRequest` keeps `None` so
the absence of evidence is visible.

## 6. Conflicting requests

Fusion evaluates approaches independently and may report several active at
once. A test feeds two simultaneous active decisions to `SignalController` and
confirms that exactly one approach receives GREEN, selected by earliest
confirmed request and then configured order. Fusion does not duplicate or
weaken the Stage 3 safety invariants.

## 7. Configuration

```yaml
fusion:
  policy: AUDIO              # VIDEO | AUDIO | BOTH | ADAPTIVE
  stale_after_s: 3.0
  debounce_s: 1.0
  adaptive_window_s: 1.0     # must be <= stale_after_s
```

All four policies are valid configuration values. The default remains `AUDIO`,
matching the original dashboard default.

### Per-approach policies

Policy selection is per approach. `FusionEngine` accepts an optional
`policies={...}` mapping at construction and exposes:

* `set_policy(approach, policy)` — change one approach, validated against the
  four policies;
* `policy_for(approach)` — read the current policy;
* `policies` — a snapshot mapping for every configured approach;
* `reset()` — clear buffered evidence, active flags and duplicate timestamps
  while retaining accepted/suppressed/expired counters and the last evaluated
  time.

`DashboardViewModel` initialises every approach from `fusion.policy`, lets each
card switch policy at runtime, and restores the configured default on
`Clear All`. Changing a policy keeps existing evidence, so a stricter policy
can take effect on the next decision cycle without waiting for new events.

## 8. Tests

Stage 7 adds 35 tests (34 in `tests/test_fusion.py` plus one configuration
test). Stage 8 adds two more tests to `tests/test_fusion.py` for per-approach
overrides and `reset()`, so that module now collects 36 tests:

* a table covering all four policies × video-only / audio-only / both / neither,
  including expected confidence and evidence IDs;
* a check that `BOTH` does not average confidences;
* staleness boundary, full expiry and partial expiry;
* ADAPTIVE being stricter than BOTH;
* duplicate suppression boundaries, different keys and out-of-order events;
* multiple fresh events, maximum confidence and full evidence IDs;
* conflicting simultaneous requests resolved by `SignalController`;
* controller/priority conversions, input validation and backwards-time rejection;
* per-approach policy overrides (independent approaches, invalid names and
  policies rejected);
* `reset()` discarding evidence and active flags without resetting counters or
  the last evaluated time.

No test claims traffic-control performance. The engine evaluates evidence
presence and confidence; it does not predict real-world safety.
