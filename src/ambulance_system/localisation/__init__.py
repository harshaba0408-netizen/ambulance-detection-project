"""Offline GCC-PHAT direction research (Stage 13).

This package is deliberately isolated from the runtime decision path. It
estimates time differences between synchronised channels and converts them to a
far-field bearing under explicit geometry assumptions. It does not classify
sirens, does not select a traffic phase, and cannot compute a geographic
position from a single microphone.

See ``docs/localisation.md`` for the derivation, synthetic results and
limitations.
"""

from __future__ import annotations

from ambulance_system.localisation.doa import (
    SPEED_OF_SOUND_M_S,
    ArrayBearingEstimate,
    DoAError,
    PairBearingEstimate,
    TdoaObservation,
    estimate_array_bearing,
    estimate_pair_bearing,
)
from ambulance_system.localisation.gcc_phat import (
    DelayEstimate,
    GccPhatError,
    estimate_delay,
    estimate_delays,
    gcc_phat_correlation,
)
from ambulance_system.localisation.simulation import (
    ArraySignals,
    TwoSourceSignals,
    fractional_delay,
    plane_wave_arrival_times,
    simulate_plane_wave,
    simulate_two_sources,
)

__all__ = [
    "SPEED_OF_SOUND_M_S",
    "ArrayBearingEstimate",
    "ArraySignals",
    "DelayEstimate",
    "DoAError",
    "GccPhatError",
    "PairBearingEstimate",
    "TdoaObservation",
    "TwoSourceSignals",
    "estimate_array_bearing",
    "estimate_delay",
    "estimate_delays",
    "estimate_pair_bearing",
    "fractional_delay",
    "gcc_phat_correlation",
    "plane_wave_arrival_times",
    "simulate_plane_wave",
    "simulate_two_sources",
]
