"""Pareto envelopes over (unmet non-critical load, metric) and comparisons on their common support.

Points are (x, y): x is unmet non-critical load, y the metric to minimise (unmet critical load, hours
above 8 C). A reserve family swept over its margin traces such points; only the efficient ones (no other
point with less or equal x and y) form its envelope. Comparisons never extrapolate: outside the common
support a challenger is classed as dominating, dominated, or with no overlap.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class Comparison:
    kind: str  # "gap", "dominates", "dominated" or "no_overlap"
    gap: float | None = None  # reference minus challenger; positive means the challenger is better
    reference: float | None = None  # reference metric over the same support
    lo: float | None = None
    hi: float | None = None


def pareto_envelope(points: list[Point]) -> list[Point]:
    """Efficient points sorted by x: each has a strictly lower y than every point with less or equal x."""
    envelope: list[Point] = []
    for x, y in sorted(points):
        if envelope and envelope[-1][0] == x:
            continue  # sorted ties: the lower y came first
        if not envelope or y < envelope[-1][1]:
            envelope.append((x, y))
    return envelope


def interpolate(envelope: list[Point], x: float) -> float | None:
    """Linear between adjacent efficient points; None outside the envelope's support."""
    if not envelope or x < envelope[0][0] or x > envelope[-1][0]:
        return None
    for (x0, y0), (x1, y1) in itertools.pairwise(envelope):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return envelope[-1][1]


def compare_point(reference: list[Point], point: Point) -> Comparison:
    x, y = point
    ref = interpolate(reference, x)
    if ref is not None:
        return Comparison("gap", ref - y, ref, x, x)
    return Comparison(_outside(reference, [point]))


def compare_frontiers(reference: list[Point], challenger: list[Point]) -> Comparison:
    """Mean vertical gap over the common support, exact for piecewise-linear envelopes."""
    lo, hi = max(reference[0][0], challenger[0][0]), min(reference[-1][0], challenger[-1][0])
    if lo > hi:
        return Comparison(_outside(reference, challenger))
    if lo == hi:
        ref = interpolate(reference, lo)
        return Comparison("gap", ref - interpolate(challenger, lo), ref, lo, hi)
    xs = sorted({lo, hi, *(x for x, _ in reference + challenger if lo < x < hi)})
    ref_area = ch_area = 0.0
    for a, b in itertools.pairwise(xs):
        ref_area += (interpolate(reference, a) + interpolate(reference, b)) / 2 * (b - a)
        ch_area += (interpolate(challenger, a) + interpolate(challenger, b)) / 2 * (b - a)
    width = hi - lo
    return Comparison("gap", (ref_area - ch_area) / width, ref_area / width, lo, hi)


def _outside(reference: list[Point], challenger: list[Point]) -> str:
    """Strict: "dominates" means some challenger point beats every reference point on both axes;
    "dominated" means the reference's right end beats every challenger point on both."""
    x_left, (x_right, y_right) = reference[0][0], reference[-1]
    if any(x < x_left and y <= y_right for x, y in challenger):
        return "dominates"
    if all(x > x_right and y >= y_right for x, y in challenger):
        return "dominated"
    return "no_overlap"
