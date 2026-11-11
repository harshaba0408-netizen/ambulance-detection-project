"""Multimodal fusion package (Stage 7)."""

from ambulance_system.fusion.engine import (
    FusionDecision,
    FusionEngine,
    FusionStats,
    FusionUpdate,
)
from ambulance_system.fusion.rules import (
    POLICY_ADAPTIVE,
    POLICY_AUDIO,
    POLICY_BOTH,
    POLICY_VIDEO,
    PolicyResult,
    evaluate_policy,
    is_fresh,
)

__all__ = [
    "FusionDecision",
    "FusionEngine",
    "FusionStats",
    "FusionUpdate",
    "POLICY_ADAPTIVE",
    "POLICY_AUDIO",
    "POLICY_BOTH",
    "POLICY_VIDEO",
    "PolicyResult",
    "evaluate_policy",
    "is_fresh",
]
