"""Summarise experiment 3: each multi-store family against the battery-only frontier; apply the pre-registered rule."""

import itertools
import json
import statistics as st
import sys
from collections import defaultdict

YEARS = 5
FAMILIES = ("multi", "cold_only", "water_only")


def interpolate(points: list[tuple[float, float]], x: float) -> float | None:
    for (x0, y0), (x1, y1) in itertools.pairwise(sorted(points)):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else min(y0, y1)
    return None


def fmt(xs: list[float]) -> str:
    return ", ".join(f"{x:+.0%}" for x in xs) or "outside frontier"


def curve(rows: list[dict], y: str) -> list[tuple[float, float, float]]:
    """(margin, mean unmet non-critical kWh/yr, mean y per year) for one family."""
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["margin"]].append(r)
    return [
        (m, st.mean(r["unmet_noncritical_kwh"] for r in rs) / YEARS, st.mean(r[y] for r in rs) / YEARS)
        for m, rs in sorted(by_m.items())
    ]


def gains(reference: list[dict], other: list[dict], y: str) -> list[float]:
    front = [(x, v) for _, x, v in curve(reference, y)]
    out = []
    for _, x, v in curve(other, y):
        ref = interpolate(front, x)
        if ref:
            out.append(1 - v / ref)
    return out


def main() -> None:
    with open(sys.argv[1]) as fh:
        rows = json.load(fh)
    g = defaultdict(list)
    for r in rows:
        g[(r["pack"], r["room"], r["family"])].append(r)
    for pack in ("new", "used"):
        print(f"\n==== {pack} pack")
        for room in ("produce", "bank", "bigbatt"):
            fams = ("batt_only",) + (FAMILIES if room != "bigbatt" else ())
            for fam in fams:
                c = curve(g[(pack, room, fam)], "hours_above_max")
                d = curve(g[(pack, room, fam)], "degree_hours_above_max")
                wear = st.mean(r["final_capacity"] for r in g[(pack, room, fam)])
                pts = "  ".join(
                    f"m{m:.2f}:{x:6.1f}/{v:5.1f}h/{dv:5.1f}Kh" for (m, x, v), (_, _, dv) in zip(c, d, strict=True)
                )
                print(f"{room:8} {fam:10} cap_end {100 * wear:5.2f}%  {pts}")
        for room in ("produce", "bank"):
            ref = g[(pack, room, "batt_only")]
            for fam in FAMILIES:
                h = gains(ref, g[(pack, room, fam)], "hours_above_max")
                dh = gains(ref, g[(pack, room, fam)], "degree_hours_above_max")
                mine = g[(pack, room, fam)]
                wear = 100 * (st.mean(r["final_capacity"] for r in ref) - st.mean(r["final_capacity"] for r in mine))
                print(f"  {room:8} {fam:10} vs batt_only: hours>8C {fmt(h)} | degree-hours {fmt(dh)}")
                print(f"  {'':8} {'':10} wear {wear:+.2f} pp")
        h = gains(g[(pack, "bigbatt", "batt_only")], g[(pack, "bank", "batt_only")], "hours_above_max")
        hm = gains(g[(pack, "bigbatt", "batt_only")], g[(pack, "bank", "multi")], "hours_above_max")
        print(f"  secondary: bank (batt_only) vs +3 kWh battery: {[f'{x:+.0%}' for x in h]}")
        print(f"  secondary: bank (multi) vs +3 kWh battery:     {[f'{x:+.0%}' for x in hm]}")


if __name__ == "__main__":
    main()
