from __future__ import annotations

import math

import pytest

from ambulance_system.clock import FakeClock
from ambulance_system.monitoring import (
    LatencyTracker,
    RateMeter,
    ResourceMonitor,
    ResourceSnapshot,
)


# ---------------------------------------------------------------------------
# RateMeter
# ---------------------------------------------------------------------------
def test_rate_meter_measures_events_per_second():
    clock = FakeClock(0.0)
    meter = RateMeter(clock)

    meter.mark(3)
    clock.advance(2.0)

    assert meter.count == 3
    assert meter.rate() == pytest.approx(1.5)


def test_rate_meter_starts_at_zero():
    meter = RateMeter(FakeClock(0.0))

    assert meter.count == 0
    assert meter.rate() == 0.0


def test_rate_meter_reset_clears_count_and_restarts_window():
    clock = FakeClock(0.0)
    meter = RateMeter(clock)
    meter.mark(4)
    clock.advance(2.0)

    meter.reset()
    assert meter.count == 0
    assert meter.rate() == 0.0

    meter.mark(2)
    clock.advance(4.0)

    assert meter.rate() == pytest.approx(0.5)


def test_rate_meter_reset_with_explicit_start():
    clock = FakeClock(0.0)
    meter = RateMeter(clock)

    meter.reset(start=-2.0)
    meter.mark(2)

    assert meter.rate() == pytest.approx(1.0)


def test_rate_meter_rejects_invalid_counts():
    meter = RateMeter(FakeClock(0.0))

    with pytest.raises(ValueError, match="count"):
        meter.mark(-1)
    with pytest.raises(ValueError, match="count"):
        meter.mark(True)


# ---------------------------------------------------------------------------
# LatencyTracker
# ---------------------------------------------------------------------------
def test_latency_tracker_reports_average_and_p95_in_milliseconds():
    tracker = LatencyTracker()

    for seconds in (0.001, 0.002, 0.003, 0.004):
        tracker.add(seconds)

    assert tracker.count == 4
    assert tracker.average_ms == pytest.approx(2.5)
    assert tracker.p95_ms == pytest.approx(4.0)


def test_latency_tracker_is_empty_before_samples():
    tracker = LatencyTracker()

    assert tracker.count == 0
    assert tracker.average_ms == 0.0
    assert tracker.p95_ms == 0.0


def test_latency_tracker_keeps_only_the_window():
    tracker = LatencyTracker(max_samples=3)

    for seconds in (0.001, 0.002, 0.003, 0.004):
        tracker.add(seconds)

    assert tracker.count == 3
    assert tracker.average_ms == pytest.approx(3.0)


def test_latency_tracker_reset_and_validation():
    tracker = LatencyTracker()

    with pytest.raises(ValueError, match="max_samples"):
        LatencyTracker(max_samples=0)

    tracker.add(0.01)
    tracker.reset()
    assert tracker.count == 0

    with pytest.raises(ValueError, match="finite"):
        tracker.add(-1.0)
    with pytest.raises(ValueError, match="finite"):
        tracker.add(float("nan"))
    with pytest.raises(ValueError, match="number"):
        tracker.add("fast")


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------
def test_resource_snapshot_serialises():
    snapshot = ResourceSnapshot(
        timestamp=10.0,
        cpu_percent=12.5,
        system_ram_percent=55.0,
        system_ram_used_mb=4096.0,
        process_ram_mb=128.0,
    )

    payload = snapshot.to_dict()

    assert payload == {
        "timestamp": 10.0,
        "cpu_percent": 12.5,
        "system_ram_percent": 55.0,
        "system_ram_used_mb": 4096.0,
        "process_ram_mb": 128.0,
    }


def test_resource_monitor_uses_injected_clock_and_reports_available_values():
    clock = FakeClock(100.0)
    monitor = ResourceMonitor(clock)

    snapshot = monitor.sample()

    assert snapshot.timestamp == pytest.approx(100.0)
    if snapshot.process_ram_mb is not None:
        assert snapshot.process_ram_mb > 0
        assert 0.0 <= snapshot.system_ram_percent <= 100.0
        assert snapshot.cpu_percent >= 0.0
    assert math.isfinite(snapshot.to_dict()["timestamp"])


def test_resource_monitor_degrades_when_psutil_is_unavailable():
    monitor = ResourceMonitor(FakeClock(5.0))
    monitor._psutil = None
    monitor._process = None

    snapshot = monitor.sample()

    assert snapshot.timestamp == pytest.approx(5.0)
    assert snapshot.cpu_percent is None
    assert snapshot.system_ram_percent is None
    assert snapshot.process_ram_mb is None
