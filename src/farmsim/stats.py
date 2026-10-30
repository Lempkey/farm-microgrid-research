"""Circular moving-block bootstrap, so that neighbouring weather years (ENSO and the like) stay together.

A resample is built from blocks of `block` consecutive years, each starting at a uniformly random year
and wrapping past the end (so the first and last years are drawn as often as the rest). Block 1 is the
ordinary iid bootstrap.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence


def resample_indices(n: int, block: int, seed: int) -> list[int]:
    if not 1 <= block <= n:
        raise ValueError(f"block length {block} must be between 1 and the series length {n}")
    rng = random.Random(seed)
    idx: list[int] = []
    while len(idx) < n:
        start = rng.randrange(n)
        idx.extend((start + k) % n for k in range(block))
    return idx[:n]


def block_bootstrap[T](
    series: Sequence[T], statistic: Callable[[list[T]], float], block: int, resamples: int, seed: int
) -> list[float]:
    """The statistic on `resamples` circular block resamples; resample i uses seed `seed + i`."""
    n = len(series)
    return [statistic([series[i] for i in resample_indices(n, block, seed + r)]) for r in range(resamples)]


def percentile_interval(samples: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    ordered = sorted(samples)
    tail = (1.0 - level) / 2
    lo = ordered[int(tail * (len(ordered) - 1))]
    hi = ordered[int(round((1.0 - tail) * (len(ordered) - 1)))]
    return lo, hi
