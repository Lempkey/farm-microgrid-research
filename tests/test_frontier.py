"""Pareto envelopes and frontier comparisons: dominated points dropped, common support only."""

import pytest

from farmsim.frontier import compare_frontiers, compare_point, interpolate, pareto_envelope

# (unmet non-critical, unmet critical): more reserve sheds more non-critical load and protects more
REFERENCE = [(10.0, 8.0), (20.0, 4.0), (40.0, 2.0)]


def test_envelope_drops_dominated_points_and_sorts_by_non_critical_loss():
    points = [(40.0, 2.0), (10.0, 8.0), (25.0, 5.0), (20.0, 4.0), (30.0, 4.0)]
    assert pareto_envelope(points) == REFERENCE


def test_envelope_of_ties_keeps_the_lower_point():
    assert pareto_envelope([(10.0, 5.0), (10.0, 3.0)]) == [(10.0, 3.0)]


def test_interpolation_only_between_adjacent_efficient_points_and_none_outside():
    assert interpolate(REFERENCE, 15.0) == pytest.approx(6.0)
    assert interpolate(REFERENCE, 30.0) == pytest.approx(3.0)
    assert interpolate(REFERENCE, 40.0) == pytest.approx(2.0)
    assert interpolate(REFERENCE, 9.9) is None
    assert interpolate(REFERENCE, 40.1) is None


def test_point_inside_support_gives_the_vertical_gap():
    c = compare_point(REFERENCE, (15.0, 4.5))
    assert c.kind == "gap"
    assert c.gap == pytest.approx(1.5)
    assert c.reference == pytest.approx(6.0)


def test_point_left_of_support_dominates_only_if_no_worse_than_every_reference_point():
    assert compare_point(REFERENCE, (5.0, 1.5)).kind == "dominates"
    assert compare_point(REFERENCE, (5.0, 7.0)).kind == "no_overlap"  # less shedding, more critical loss


def test_point_right_of_support_and_no_better_than_its_right_end_is_dominated():
    assert compare_point(REFERENCE, (50.0, 3.0)).kind == "dominated"
    assert compare_point(REFERENCE, (50.0, 1.0)).kind == "no_overlap"


def test_frontier_gap_is_the_exact_mean_over_common_support():
    challenger = [(15.0, 4.0), (40.0, 1.0)]  # straight line y = 4 - 0.12 (x - 15)
    c = compare_frontiers(REFERENCE, challenger)
    assert c.kind == "gap"
    assert (c.lo, c.hi) == (15.0, 40.0)
    # reference mean over [15, 40]: (5 * 5 + 20 * 3) / 25 = 3.4; challenger mean: 2.5
    assert c.reference == pytest.approx(3.4)
    assert c.gap == pytest.approx(0.9)


def test_identical_frontiers_have_zero_gap():
    c = compare_frontiers(REFERENCE, REFERENCE)
    assert c.gap == pytest.approx(0.0)
    assert c.reference == pytest.approx(compare_frontiers(REFERENCE, REFERENCE).reference)


def test_frontier_wholly_left_and_lower_dominates():
    assert compare_frontiers(REFERENCE, [(2.0, 3.0), (6.0, 1.0)]).kind == "dominates"


def test_frontier_wholly_right_and_higher_is_dominated():
    assert compare_frontiers(REFERENCE, [(45.0, 3.0), (60.0, 2.5)]).kind == "dominated"


def test_single_point_overlap_compares_at_that_point():
    c = compare_frontiers(REFERENCE, [(40.0, 1.0), (60.0, 0.5)])
    assert c.kind == "gap"
    assert (c.lo, c.hi) == (40.0, 40.0)
    assert c.gap == pytest.approx(1.0)
