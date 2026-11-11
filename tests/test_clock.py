from __future__ import annotations

import time

import pytest

from ambulance_system.clock import Clock, FakeClock, RealClock


def test_fake_clock_starts_at_zero_by_default():
    clock = FakeClock()

    assert clock.now() == 0.0
    assert clock.monotonic() == 0.0


def test_fake_clock_advances_both_readings():
    clock = FakeClock(start=100.0)
    returned = clock.advance(2.5)

    assert returned == pytest.approx(102.5)
    assert clock.now() == pytest.approx(102.5)
    assert clock.monotonic() == pytest.approx(102.5)


def test_fake_clock_supports_independent_monotonic_start():
    clock = FakeClock(start=100.0, monotonic_start=5.0)

    assert clock.now() == pytest.approx(100.0)
    assert clock.monotonic() == pytest.approx(5.0)


def test_fake_clock_rejects_negative_advance():
    clock = FakeClock()

    with pytest.raises(ValueError, match="seconds"):
        clock.advance(-1.0)


def test_fake_clock_rejects_non_finite_advance():
    clock = FakeClock()

    with pytest.raises(ValueError, match="seconds"):
        clock.advance(float("nan"))


def test_real_clock_satisfies_protocol_and_advances():
    clock = RealClock()
    first = clock.monotonic()
    time.sleep(0.01)
    second = clock.monotonic()

    assert isinstance(clock, Clock)
    assert second >= first
    assert clock.now() > 0.0
