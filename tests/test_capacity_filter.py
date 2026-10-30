"""Capacity filter: a lone outlier is held, two consistent ones confirm a step (pre-registered in R-48 A3)."""

import math

import pytest

from farmsim.healthsim import (
    CAPACITY_BOUNDS,
    JUMP_RESET_SIGMA,
    CapacityFilter,
    filter_predict,
    filter_update,
)

R = 0.01**2  # pair sample variance, ~1 pp
CONVERGED = CapacityFilter(est=0.90, var=0.003**2)


def test_sample_within_four_sigma_is_a_standard_update():
    f, event, _ = filter_update(CONVERGED, z=0.91, r=R, day=0)
    gain = CONVERGED.var / (CONVERGED.var + R)
    assert event == "update"
    assert f.est == pytest.approx(0.90 + gain * 0.01)
    assert f.candidate is None


def test_lone_outlier_is_held_and_leaves_the_estimate_unchanged():
    f, event, _ = filter_update(CONVERGED, z=0.75, r=R, day=10)
    assert event == "held"
    assert (f.est, f.var) == (CONVERGED.est, CONVERGED.var)
    f, event, _ = filter_update(f, z=0.90, r=R, day=12)
    assert event == "update"
    assert f.candidate is None
    assert f.est == pytest.approx(0.90, abs=1e-3)


def test_true_step_is_absorbed_on_the_second_consistent_sample():
    f, _, _ = filter_update(CONVERGED, z=0.752, r=R, day=10)
    f, event, _ = filter_update(f, z=0.748, r=R, day=12)
    assert event == "jump"
    assert f.est == pytest.approx(0.75)
    assert f.var == pytest.approx(max(R / 2, JUMP_RESET_SIGMA**2))  # post-jump floor, not R/2
    assert f.candidate is None


def test_false_confirmation_by_two_biased_samples_recovers_within_two_normal_samples():
    f, _, _ = filter_update(CONVERGED, z=0.75, r=R, day=10)
    f, event, _ = filter_update(f, z=0.75, r=R, day=11)
    assert event == "jump"  # the documented failure mode: two consistent outliers are believed
    events = []
    for day in (13, 15):
        f, event, _ = filter_update(f, z=0.90, r=R, day=day)
        events.append(event)
    assert events == ["held", "jump"]
    assert f.est == pytest.approx(0.90)


def test_candidate_expires_after_thirty_days():
    f, _, _ = filter_update(CONVERGED, z=0.75, r=R, day=0)
    f, event, _ = filter_update(f, z=0.75, r=R, day=31)
    assert event == "held"  # the first one expired, this one is the new candidate
    assert f.candidate.day == 31
    f, event, _ = filter_update(f, z=0.75, r=R, day=33)
    assert event == "jump"


def test_inconsistent_or_opposite_outlier_replaces_the_candidate():
    f, _, _ = filter_update(CONVERGED, z=0.75, r=R, day=0)
    f, event, _ = filter_update(f, z=0.60, r=R, day=2)  # same sign, 15 pp apart
    assert event == "held"
    assert f.candidate.z == 0.60
    f, event, _ = filter_update(f, z=1.00, r=R, day=4)  # opposite sign
    assert event == "held"
    assert f.candidate.sign == 1
    assert f.est == CONVERGED.est


def test_noisier_samples_widen_the_consistency_window():
    noisy = 0.04**2
    f, _, _ = filter_update(CONVERGED, z=0.70, r=noisy, day=0)
    f, event, _ = filter_update(f, z=0.62, r=noisy, day=3)  # 8 pp apart, within 3 * sqrt(2) * 4 pp
    assert event == "jump"
    assert f.est == pytest.approx(0.66)


def test_slow_drift_is_tracked_without_triggering_the_jump_rule():
    f, truth, events = CapacityFilter(est=1.0, var=0.01**2), 1.0, []
    for day in range(3 * 365):
        truth -= 0.02 / 365
        f, _ = filter_predict(f, drift=0.02 / 365, process_var=0.0005**2)
        if day % 3 == 0:
            noise = 0.01 * math.sin(day)  # bounded, deterministic noise of up to 1 pp
            f, event, _ = filter_update(f, z=truth + noise, r=R, day=day)
            events.append(event)
    assert "held" not in events
    assert f.est == pytest.approx(truth, abs=0.01)


def test_estimate_is_clipped_to_physical_bounds_after_updates_jumps_and_predictions():
    lo, hi = CAPACITY_BOUNDS
    f, _, _ = filter_update(CapacityFilter(est=1.0, var=0.01**2), z=1.30, r=R, day=0)
    f, event, clipped = filter_update(f, z=1.30, r=R, day=1)
    assert event == "jump"
    assert clipped
    assert f.est == hi
    f, clipped = filter_predict(CapacityFilter(est=lo + 0.001, var=R), drift=0.01, process_var=0.0)
    assert clipped
    assert f.est == lo
