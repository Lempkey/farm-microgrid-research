"""Circular moving-block bootstrap over weather years."""

import statistics as st
from collections import Counter

import pytest

from farmsim.stats import block_bootstrap, percentile_interval, resample_indices

YEARS = list(range(24))


def test_resample_has_the_series_length_and_is_reproducible():
    a = resample_indices(24, block=2, seed=1)
    assert len(a) == 24
    assert a == resample_indices(24, block=2, seed=1)


def test_blocks_are_consecutive_years_wrapping_around_the_end():
    idx = resample_indices(24, block=3, seed=5)
    for k in range(0, 24, 3):
        a, b, c = idx[k : k + 3]
        assert b == (a + 1) % 24 and c == (b + 1) % 24


def test_circular_blocks_use_every_year_equally_often_including_the_ends():
    counts = Counter()
    for seed in range(4000):
        counts.update(resample_indices(24, block=3, seed=seed))
    expected = 4000
    assert all(abs(counts[i] - expected) / expected < 0.06 for i in range(24))


def test_block_of_one_is_the_iid_bootstrap():
    idx = resample_indices(24, block=1, seed=2)
    assert len(set(idx)) < 24  # draws with replacement


def test_bootstrap_of_a_constant_series_has_a_degenerate_interval():
    samples = block_bootstrap([3.0] * 24, st.mean, block=2, resamples=200, seed=0)
    assert percentile_interval(samples) == (3.0, 3.0)


def test_bootstrap_mean_is_centred_on_the_sample_mean():
    series = [float(i % 7) for i in YEARS]
    samples = block_bootstrap(series, st.mean, block=2, resamples=5000, seed=3)
    assert st.mean(samples) == pytest.approx(st.mean(series), abs=0.05)
    lo, hi = percentile_interval(samples)
    assert lo < st.mean(series) < hi


def test_statistic_can_use_paired_values():
    pairs = [(1.0, 4.0)] * 24  # (gap, reference): ratio of sums
    samples = block_bootstrap(pairs, lambda xs: sum(g for g, _ in xs) / sum(r for _, r in xs), 2, 50, 0)
    assert set(samples) == {0.25}


def test_block_longer_than_the_series_is_refused():
    with pytest.raises(ValueError):
        resample_indices(3, block=4, seed=0)
