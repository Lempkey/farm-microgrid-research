"""The per-year summary keeps every year, in order, when it bootstraps (regression test for the PR #3 review)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from summarise_per_year import _interval, _mean_gap, _ratio, outcome_counts, summarise  # noqa: E402

from farmsim.frontier import Comparison  # noqa: E402

SUPPORTED_GAP_SUPPORTED = [(1.0, 2.0), None, (3.0, 4.0)]  # 2002 has no common support


def test_statistics_use_only_the_supported_years():
    assert _mean_gap(SUPPORTED_GAP_SUPPORTED) == pytest.approx(2.0)
    assert _ratio(SUPPORTED_GAP_SUPPORTED) == pytest.approx(4.0 / 6.0)


def test_bootstrap_keeps_the_unsupported_year_in_place():
    # A circular block of 3 over 3 years always holds all three, so the interval collapses to the
    # full-sample ratio. Dropping the unsupported year would leave 2 years, and a block of 3 is refused.
    assert _interval(SUPPORTED_GAP_SUPPORTED, _ratio, block=3) == (pytest.approx(4 / 6), pytest.approx(4 / 6))


def test_summary_accounts_for_every_year_and_keeps_awkward_ones_in_the_resampling():
    comparisons = [
        (2001, Comparison("gap", 1.0, 2.0)),
        (2002, Comparison("dominated")),
        (2003, Comparison("gap", 3.0, 4.0)),
        (2004, Comparison("dominates")),
        (2005, Comparison("gap", -0.5, 2.0)),
        (2006, Comparison("gap", 0.0, 0.0)),
    ]
    s = summarise(comparisons)
    assert len(s.years) == 6
    assert outcome_counts(s) == (
        "dominates 1, better on common support 2, tied 1, worse on common support 1, dominated 1, no overlap 0"
        " (of 6 years)"
    )
    assert s.mean_gap == pytest.approx(3.5 / 4)
    assert s.ratio == pytest.approx(3.5 / 8)
    lo, hi = s.ratio_ci["block 2"]
    assert lo <= s.ratio <= hi
