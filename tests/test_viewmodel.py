"""Headless tests for the Stage 8 dashboard view-model.

These tests never import Tkinter. They drive the same evidence, policies and
decision cycle used by the real dashboard, so UI behaviour can be verified on a
machine without a display.
"""

from __future__ import annotations

import threading

import pytest

from ambulance_system.clock import FakeClock
from ambulance_system.config import AppConfig, FusionConfig, SignalConfig, SystemConfig
from ambulance_system.dashboard import DashboardViewModel

APPROACHES = ("Road 1", "Road 2", "Road 3", "Road 4")


def make_config(
    policy: str = "AUDIO",
    *,
    stale: float = 3.0,
    debounce: float = 1.0,
    adaptive: float = 1.0,
    min_confirmation: float = 1.0,
    release_grace: float = 2.0,
    all_red: float = 1.0,
    max_priority: float = 30.0,
) -> AppConfig:
    return AppConfig(
        signal=SignalConfig(
            activation_threshold=0.65,
            release_threshold=0.50,
            min_confirmation_s=min_confirmation,
            release_grace_s=release_grace,
            max_priority_s=max_priority,
            min_green_s=2.0,
            yellow_s=1.0,
            all_red_s=all_red,
        ),
        fusion=FusionConfig(
            policy=policy,
            stale_after_s=stale,
            debounce_s=debounce,
            adaptive_window_s=adaptive,
        ),
        system=SystemConfig(num_approaches=4, approach_names=APPROACHES),
    )


def make_vm(policy: str = "AUDIO", **kwargs):
    clock = FakeClock(0.0)
    viewmodel = DashboardViewModel(make_config(policy, **kwargs), clock=clock)
    viewmodel.start()
    return viewmodel, clock


# ---------------------------------------------------------------------------
# Initial state and policies
# ---------------------------------------------------------------------------
def test_initial_snapshot_is_all_red_and_running():
    vm, _ = make_vm()
    snapshot = vm.snapshot()

    assert snapshot.running is True
    assert snapshot.active_approach is None
    assert snapshot.fault is False
    assert len(snapshot.approaches) == 4
    assert [view.name for view in snapshot.approaches] == list(APPROACHES)
    for view in snapshot.approaches:
        assert view.colour == "red"
        assert view.phase == "normal"
        assert view.policy == "AUDIO"
        assert view.fused_active is False
        assert view.fused_reason == "waiting for evaluation"
        assert view.audio.active is False
        assert view.video.active is False
        assert view.bbox is None


def test_policies_are_per_approach_and_validated():
    vm, _ = make_vm()
    vm.set_policy("Road 2", "VIDEO")
    vm.set_policy("Road 3", "ADAPTIVE")
    vm.set_policy("Road 4", "BOTH")

    snapshot = vm.snapshot()
    assert snapshot.approach("Road 1").policy == "AUDIO"
    assert snapshot.approach("Road 2").policy == "VIDEO"
    assert snapshot.approach("Road 3").policy == "ADAPTIVE"
    assert snapshot.approach("Road 4").policy == "BOTH"

    with pytest.raises(ValueError, match="unknown fusion policy"):
        vm.set_policy("Road 1", "TURBO")
    with pytest.raises(ValueError, match="unknown approach"):
        vm.set_policy("Road 9", "VIDEO")
    with pytest.raises(ValueError, match="unknown approach"):
        vm.policy_for("Road 9")


def test_policy_change_keeps_existing_evidence():
    vm, clock = make_vm("BOTH")
    clock.advance(1.0)
    vm.observe_video("Road 1", 0.9, detected=True, bbox=(1, 2, 3, 4))

    assert vm.tick().approach("Road 1").fused_active is False
    vm.set_policy("Road 1", "VIDEO")
    assert vm.tick().approach("Road 1").fused_active is True


# ---------------------------------------------------------------------------
# Evidence, confidence and confirmation
# ---------------------------------------------------------------------------
def test_video_evidence_needs_confirmation_then_grants_one_green():
    vm, clock = make_vm("VIDEO")
    clock.advance(1.0)
    accepted = vm.observe_video(
        "Road 1", 0.90, detected=True, bbox=(10, 20, 30, 40), frame_index=7
    )
    assert accepted is True

    first = vm.tick()
    road = first.approach("Road 1")
    assert road.fused_active is True
    assert road.fused_confidence == pytest.approx(0.90)
    assert road.video.active is True
    assert road.video.confidence == pytest.approx(0.90)
    assert road.bbox == (10.0, 20.0, 30.0, 40.0)
    assert road.colour == "red"  # confirmation + all-red interval not complete

    clock.advance(1.0)
    second = vm.tick()
    road = second.approach("Road 1")
    assert road.colour == "green"
    assert road.phase == "emergency_priority"
    assert second.active_approach == "Road 1"
    assert [item.colour for item in second.approaches].count("green") == 1


def test_audio_policy_uses_audio_and_ignores_video():
    vm, clock = make_vm("AUDIO")
    clock.advance(1.0)
    vm.observe_video("Road 1", 0.95, detected=True)
    assert vm.tick().approach("Road 1").fused_active is False

    clock.advance(1.0)
    assert vm.observe_audio(
        "Road 1", 0.80, is_siren=True, window_start_s=0.0, window_end_s=3.0
    ) is True
    snapshot = vm.tick()
    road = snapshot.approach("Road 1")
    assert road.fused_active is True
    assert road.fused_confidence == pytest.approx(0.80)
    assert road.audio.active is True
    assert road.video.active is True  # last video evidence is still fresh
    assert road.audio.confidence == pytest.approx(0.80)


def test_audio_hard_label_without_probability_is_preserved():
    vm, clock = make_vm("AUDIO")
    clock.advance(1.0)
    vm.observe_audio("Road 2", None, is_siren=True)
    snapshot = vm.tick()
    assert snapshot.approach("Road 2").fused_confidence == pytest.approx(1.0)


def test_both_policy_requires_both_modalities_and_uses_conservative_minimum():
    vm, clock = make_vm("BOTH")
    clock.advance(1.0)
    vm.observe_video("Road 1", 0.90, detected=True)
    assert vm.tick().approach("Road 1").fused_active is False

    clock.advance(1.0)
    vm.observe_audio("Road 1", 0.70, is_siren=True)
    snapshot = vm.tick()
    road = snapshot.approach("Road 1")
    assert road.fused_active is True
    assert road.fused_confidence == pytest.approx(0.70)  # min, never averaged

    clock.advance(1.0)
    assert vm.tick().approach("Road 1").colour == "green"


def test_adaptive_requires_both_inside_the_tighter_window():
    vm, clock = make_vm("ADAPTIVE", stale=5.0, adaptive=1.0)
    clock.advance(1.0)
    vm.observe_video("Road 1", 0.90, detected=True)

    clock.advance(2.0)  # video age 2.0 > adaptive window 1.0
    vm.observe_audio("Road 1", 0.80, is_siren=True)
    assert vm.tick().approach("Road 1").fused_active is False

    clock.advance(1.0)
    vm.observe_video("Road 1", 0.90, detected=True)  # fresh video inside window
    assert vm.tick().approach("Road 1").fused_active is True


def test_duplicate_evidence_is_suppressed_but_display_still_updates():
    vm, clock = make_vm("VIDEO")
    clock.advance(1.0)
    assert vm.observe_video("Road 1", 0.8, detected=True) is True
    assert vm.observe_video("Road 1", 0.9, detected=True) is False
    assert vm.tick().approach("Road 1").video.confidence == pytest.approx(0.9)


def test_video_staleness_clears_fused_state_and_bbox():
    vm, clock = make_vm("VIDEO", stale=3.0)
    clock.advance(1.0)
    vm.observe_video("Road 1", 0.9, detected=True, bbox=(1, 1, 5, 5))
    assert vm.tick().approach("Road 1").fused_active is True

    clock.advance(3.5)
    snapshot = vm.tick()
    road = snapshot.approach("Road 1")
    assert road.fused_active is False
    assert road.video.active is False
    assert road.bbox is None


# ---------------------------------------------------------------------------
# Signal lifecycle
# ---------------------------------------------------------------------------
def test_clear_all_stops_priority_and_restores_default_policies():
    vm, clock = make_vm("VIDEO")
    vm.set_policy("Road 2", "VIDEO")
    clock.advance(1.0)
    vm.observe_video("Road 2", 0.9, detected=True)
    vm.tick()
    clock.advance(1.0)
    assert vm.tick().approach("Road 2").colour == "green"

    clock.advance(1.0)
    snapshot = vm.clear_all()

    assert snapshot.active_approach is None
    assert snapshot.fault is False
    for view in snapshot.approaches:
        assert view.colour == "red"
        assert view.phase == "normal"
        assert view.fused_active is False
        assert view.video.active is False
        assert view.video_fps == 0.0
    assert snapshot.approach("Road 2").policy == "VIDEO"
    assert snapshot.events[-1].message == "manual Clear All"


def test_fault_forces_all_red_and_needs_explicit_recovery():
    vm, clock = make_vm("VIDEO")
    faulted = vm.trigger_fault("test fault")
    assert faulted.fault is True
    for view in faulted.approaches:
        assert view.colour == "red"
        assert view.phase == "fault"

    clock.advance(5.0)
    still_faulted = vm.tick()
    assert still_faulted.fault is True

    recovered = vm.clear_fault()
    assert recovered.fault is False
    assert all(view.phase == "normal" for view in recovered.approaches)


# ---------------------------------------------------------------------------
# Metrics, event history and thread safety
# ---------------------------------------------------------------------------
def test_health_and_video_metrics_are_exposed():
    vm, _ = make_vm()
    vm.update_video_stats(
        "Road 1", fps=12.5, latency_ms=345.0, frames_dropped=9, last_error="boom"
    )
    vm.update_health(cpu_percent=42.0, system_ram_percent=71.0, process_ram_mb=512.0)
    snapshot = vm.snapshot()
    road = snapshot.approach("Road 1")

    assert road.video_fps == pytest.approx(12.5)
    assert road.video_latency_ms == pytest.approx(345.0)
    assert road.frames_dropped == 9
    assert road.video_error == "boom"
    assert snapshot.health.cpu_percent == pytest.approx(42.0)
    assert snapshot.health.system_ram_percent == pytest.approx(71.0)
    assert snapshot.health.process_ram_mb == pytest.approx(512.0)


def test_event_history_is_bounded_sequenced_and_marks_errors():
    vm, _ = make_vm()
    vm.stop()  # avoid mixing the "Dashboard started" event with this assertion
    for index in range(8):
        vm.log_event("INFO" if index < 7 else "ERROR", f"event {index}")

    snapshot = vm.snapshot()
    assert len(snapshot.events) <= 200
    sequences = [entry.sequence for entry in snapshot.events]
    assert sequences == sorted(sequences)
    assert sequences[-1] > sequences[0]
    assert snapshot.status_message == "event 7"
    assert snapshot.status_level == "ERROR"
    assert snapshot.health.errors == 1

    with pytest.raises(ValueError, match="level"):
        vm.log_event("VERBOSE", "nope")
    with pytest.raises(ValueError, match="message"):
        vm.log_event("INFO", "  ")


def test_event_history_cap_keeps_the_newest_entries():
    clock = FakeClock(0.0)
    vm = DashboardViewModel(make_config(), clock=clock, event_history=5)
    for index in range(8):
        vm.log_event("INFO", f"event {index}")
    snapshot = vm.snapshot()
    assert [entry.message for entry in snapshot.events] == [
        f"event {index}" for index in range(3, 8)
    ]


def test_workers_can_log_and_submit_concurrently():
    vm, clock = make_vm("AUDIO", stale=10.0)

    def worker(index: int) -> None:
        for step in range(25):
            vm.log_event("INFO", f"worker {index} step {step}", "Road 1")
        vm.observe_audio("Road 1", 0.8, is_siren=True, timestamp=1.0 + index * 2.0)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    clock.advance(20.0)
    snapshot = vm.tick()
    assert snapshot.approach("Road 1").audio.active is True
    assert len(snapshot.events) == 152  # start + 150 logs + one fusion state change


def test_snapshot_is_immutable_and_independent():
    vm, clock = make_vm("VIDEO")
    clock.advance(1.0)
    before = vm.tick()
    clock.advance(1.0)
    vm.observe_video("Road 1", 0.9, detected=True)
    after = vm.tick()

    assert before.timestamp != after.timestamp
    assert before.approach("Road 1").fused_active is False
    assert after.approach("Road 1").fused_active is True
    assert isinstance(after.approaches, tuple)
    with pytest.raises(Exception):
        after.status_message = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    "call",
    [
        lambda vm: vm.observe_video("Road 9", 0.9, detected=True),
        lambda vm: vm.observe_audio("Road 9", 0.9, is_siren=True),
        lambda vm: vm.observe_video("Road 1", 1.5, detected=True),
        lambda vm: vm.observe_video("Road 1", None, detected=True),
        lambda vm: vm.observe_video("Road 1", 0.9, detected="yes"),
        lambda vm: vm.observe_audio("Road 1", 0.9, is_siren="yes"),
        lambda vm: vm.observe_video("Road 1", 0.9, detected=True, timestamp=-1.0),
        lambda vm: vm.update_video_stats("Road 1", fps=-2.0),
        lambda vm: vm.update_video_stats("Road 1", frames_dropped=-1),
        lambda vm: vm.update_health(cpu_percent=float("nan")),
    ],
)
def test_invalid_inputs_are_rejected(call):
    vm, _ = make_vm()
    with pytest.raises(ValueError):
        call(vm)
