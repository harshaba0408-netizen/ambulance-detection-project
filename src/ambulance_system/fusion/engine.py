"""Stateful multimodal fusion engine.

The engine ingests `DetectionEvent` objects, suppresses duplicates, expires
stale evidence and applies one configured policy per approach. The controller
in `traffic_signal` still performs cross-approach arbitration; fusion only says
whether an approach currently has an emergency.

Design choices:

* Confidence is never averaged across modalities.
* Duplicate suppression uses `(approach, source, detection_type)` and event
  timestamps, so replaying the same event twice does not count twice.
* Evidence expires after `fusion.stale_after_s`.
* An update reports a state change only when an approach's `active` flag flips.
  Each change carries the IDs of the evidence that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ambulance_system.clock import Clock, RealClock
from ambulance_system.config import VALID_POLICIES, FusionConfig
from ambulance_system.events import DetectionEvent, EventSource, PriorityRequest
from ambulance_system.fusion.rules import PolicyResult, evaluate_policy
from ambulance_system.logging_utils import get_logger
from ambulance_system.traffic_signal.states import ApproachObservation


@dataclass(frozen=True)
class FusionDecision:
    """One approach's fused state for one decision cycle."""

    approach: str
    active: bool
    confidence: Optional[float]
    reason: str
    evidence_ids: tuple[str, ...]
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.approach, str) or not self.approach.strip():
            raise ValueError("approach must be a non-empty string")
        if not isinstance(self.active, bool):
            raise ValueError("active must be a bool")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be within [0.0, 1.0] or None")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not isinstance(self.evidence_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence_ids
        ):
            raise ValueError("evidence_ids must be a tuple of non-empty strings")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
            or float(self.timestamp) < 0.0
        ):
            raise ValueError("timestamp must be finite and >= 0")

    def to_observation(self) -> ApproachObservation:
        """Convert to the Stage 3 controller input."""
        return ApproachObservation(
            approach=self.approach,
            requested=self.active,
            confidence=0.0 if self.confidence is None else float(self.confidence),
            note=self.reason,
        )

    def to_priority_request(self) -> PriorityRequest:
        return PriorityRequest(
            approach=self.approach,
            active=self.active,
            reason=self.reason,
            timestamp=float(self.timestamp),
            event_ids=self.evidence_ids,
            confidence=None if self.confidence is None else float(self.confidence),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "approach": self.approach,
            "active": self.active,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "timestamp": float(self.timestamp),
        }


@dataclass(frozen=True)
class FusionUpdate:
    """Result of one full evaluation cycle."""

    decisions: dict[str, FusionDecision]
    changed: tuple[FusionDecision, ...]

    def to_observations(self) -> dict[str, ApproachObservation]:
        return {
            name: decision.to_observation()
            for name, decision in self.decisions.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "decisions": [decision.to_dict() for decision in self.decisions.values()],
            "changed": [decision.to_dict() for decision in self.changed],
        }


@dataclass(frozen=True)
class FusionStats:
    """Ingest and expiry counters, useful for logs and tests."""

    accepted: int
    suppressed: int
    expired: int
    evaluations: int


class FusionEngine:
    """Per-approach policy evaluation with temporal filtering."""

    def __init__(
        self,
        config: FusionConfig,
        approaches: Sequence[str],
        *,
        policies: Optional[Mapping[str, str]] = None,
        clock: Optional[Clock] = None,
        logger=None,
    ) -> None:
        names = tuple(approaches)
        if not names:
            raise ValueError("at least one approach is required")
        if len(set(names)) != len(names):
            raise ValueError("approach names must be unique")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("approach names must be non-empty strings")

        self._config = config
        self._approaches = names
        self._clock = clock or RealClock()
        self._logger = logger or get_logger("fusion")
        self._policies: dict[str, str] = {name: config.policy for name in names}
        if policies is not None:
            if not isinstance(policies, Mapping):
                raise TypeError("policies must be a mapping of approach -> policy")
            unknown = set(policies) - set(names)
            if unknown:
                raise ValueError(f"unknown approach(es): {', '.join(sorted(unknown))}")
            for name, policy in policies.items():
                if policy not in VALID_POLICIES:
                    raise ValueError(f"unknown fusion policy: {policy!r}")
                self._policies[name] = policy

        self._events: dict[str, list[DetectionEvent]] = {name: [] for name in names}
        self._last_submitted: dict[tuple[str, EventSource, object], float] = {}
        self._active: dict[str, bool] = {name: False for name in names}
        self._last_now: Optional[float] = None
        self._accepted = 0
        self._suppressed = 0
        self._expired = 0
        self._evaluations = 0

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------
    @property
    def policy(self) -> str:
        return self._config.policy

    @property
    def policies(self) -> dict[str, str]:
        """Current policy for every approach (a copy)."""
        return dict(self._policies)

    def policy_for(self, approach: str) -> str:
        """Return the policy currently applied to one approach."""
        self._require_approach(approach)
        return self._policies[approach]

    def set_policy(self, approach: str, policy: str) -> None:
        """Change one approach's policy without discarding stored evidence."""
        self._require_approach(approach)
        if policy not in VALID_POLICIES:
            raise ValueError(f"unknown fusion policy: {policy!r}")
        self._policies[approach] = policy

    @property
    def approaches(self) -> tuple[str, ...]:
        return self._approaches

    @property
    def active_approaches(self) -> tuple[str, ...]:
        return tuple(name for name in self._approaches if self._active[name])

    @property
    def stats(self) -> FusionStats:
        return FusionStats(
            accepted=self._accepted,
            suppressed=self._suppressed,
            expired=self._expired,
            evaluations=self._evaluations,
        )

    def reset(self) -> None:
        """Discard all evidence and active flags.

        Operational counters are retained because they describe how much work
        the engine has done, not what it currently believes. The last evaluated
        timestamp is also retained so time still cannot move backwards.
        """
        for name in self._approaches:
            self._events[name].clear()
            self._active[name] = False
        self._last_submitted.clear()

    # ------------------------------------------------------------------
    # Evidence ingest
    # ------------------------------------------------------------------
    def submit(self, event: DetectionEvent) -> bool:
        """Accept one event, or suppress it as a duplicate.

        Returns True when the event was stored, False when the debounce window
        suppressed it.
        """
        if not isinstance(event, DetectionEvent):
            raise TypeError("event must be a DetectionEvent")
        if event.approach not in self._events:
            raise ValueError(f"unknown approach: {event.approach!r}")

        key = (event.approach, event.source, event.detection_type)
        previous = self._last_submitted.get(key)
        if previous is not None and float(event.timestamp) - previous < self._config.debounce_s:
            self._suppressed += 1
            self._logger.debug(
                "fusion duplicate suppressed",
                extra={
                    "event": {
                        "approach": event.approach,
                        "source": event.source.value,
                        "detection_type": event.detection_type.value,
                        "timestamp": float(event.timestamp),
                    }
                },
            )
            return False

        self._last_submitted[key] = float(event.timestamp)
        self._events[event.approach].append(event)
        self._accepted += 1
        return True

    # ------------------------------------------------------------------
    # Decision cycle
    # ------------------------------------------------------------------
    def evaluate(self, now: Optional[float] = None) -> FusionUpdate:
        """Evaluate every approach and return current + changed decisions."""
        resolved = float(self._clock.now() if now is None else now)
        if not math.isfinite(resolved) or resolved < 0.0:
            raise ValueError("now must be finite and >= 0")
        if self._last_now is not None and resolved < self._last_now:
            raise ValueError(
                f"time cannot move backwards (last={self._last_now:.6f}, now={resolved:.6f})"
            )

        decisions: dict[str, FusionDecision] = {}
        changed: list[FusionDecision] = []
        for name in self._approaches:
            fresh = self._prune(name, resolved)
            result: PolicyResult = evaluate_policy(
                self._policies[name],
                fresh,
                now=resolved,
                adaptive_window_s=self._config.adaptive_window_s,
            )
            decision = FusionDecision(
                approach=name,
                active=result.active,
                confidence=result.confidence,
                reason=result.reason,
                evidence_ids=result.evidence_ids,
                timestamp=resolved,
            )
            decisions[name] = decision
            if decision.active != self._active[name]:
                self._active[name] = decision.active
                changed.append(decision)
                self._logger.info(
                    "fusion state change",
                    extra={"event": decision.to_dict(), "approach": name},
                )

        self._last_now = resolved
        self._evaluations += 1
        return FusionUpdate(decisions=decisions, changed=tuple(changed))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_approach(self, approach: str) -> None:
        if approach not in self._events:
            raise ValueError(f"unknown approach: {approach!r}")

    def _prune(self, approach: str, now: float) -> list[DetectionEvent]:
        kept: list[DetectionEvent] = []
        expired = 0
        for event in self._events[approach]:
            if now - float(event.timestamp) <= self._config.stale_after_s:
                kept.append(event)
            else:
                expired += 1
        if expired:
            self._events[approach] = kept
            self._expired += expired
        return kept
