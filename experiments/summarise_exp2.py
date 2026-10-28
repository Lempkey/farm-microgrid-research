"""Summarise experiment 2: forecast-sized reserve against the fixed-reserve frontier; apply the pre-registered rule."""

import json
import statistics as st
import sys
from collections import defaultdict

YEARS = 5
FIXED = ["fixed_05", "fixed_20", "fixed_35", "fixed_50", "fixed_65", "fixed_80"]
FORECAST = ["fc_z0", "fc_z1", "fc_z2", "fc_climatology", "fc_perfect"]


def frontier_critical(points: list[tuple[float, float]], nc: float) -> float | None:
    """Linear interpolation of unmet critical at unmet non-critical `nc` along the fixed-reserve frontier."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if x0 <= nc <= x1:
            return y0 + (y1 - y0) * (nc - x0) / (x1 - x0) if x1 > x0 else min(y0, y1)
    return None


rows = json.load(open(sys.argv[1]))
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    agg[(r["farm"], r["pack"])][r["policy"]].append(r)

for key in sorted(agg):
    pols = agg[key]

    def m(pol: str, field: str, pols: dict = pols) -> float:
        return st.mean(r[field] for r in pols[pol])

    print(f"\n==== {key[0]} sizing, {key[1]} pack")
    print("policy           unmetC/yr  unmetNC/yr  cap_end%  served%")
    for pol in FIXED + FORECAST:
        served = 100 * m(pol, "served_kwh") / m(pol, "demand_kwh")
        print(
            f"{pol:15} {m(pol, 'unmet_critical_kwh') / YEARS:9.1f} {m(pol, 'unmet_noncritical_kwh') / YEARS:11.1f}"
            f" {100 * m(pol, 'final_capacity'):9.2f} {served:8.2f}"
        )
    front = [(m(p, "unmet_noncritical_kwh"), m(p, "unmet_critical_kwh")) for p in FIXED]
    seeds = sorted({r["seed"] for r in pols["fixed_35"]})
    for pol in FORECAST:
        nc, c = m(pol, "unmet_noncritical_kwh"), m(pol, "unmet_critical_kwh")
        ref = frontier_critical(front, nc)
        gain = None if ref is None or ref == 0 else 1 - c / ref
        per_seed = []
        for s in seeds:
            fs = [
                (
                    next(r for r in pols[p] if r["seed"] == s)["unmet_noncritical_kwh"],
                    next(r for r in pols[p] if r["seed"] == s)["unmet_critical_kwh"],
                )
                for p in FIXED
            ]
            me = next(r for r in pols[pol] if r["seed"] == s)
            rf = frontier_critical(fs, me["unmet_noncritical_kwh"])
            if rf:
                per_seed.append(1 - me["unmet_critical_kwh"] / rf)
        wear = 100 * (m("fixed_35", "final_capacity") - m(pol, "final_capacity"))
        g = "outside frontier" if gain is None else f"{gain:+.0%}"
        better = sum(x > 0 for x in per_seed)
        med = f"{st.median(per_seed):+.0%}" if per_seed else "n/a"
        print(
            f"  {pol:15} vs frontier at equal non-critical loss: {g};"
            f" per-seed median {med}, better in {better}/{len(per_seed)}; wear vs fixed_35 {wear:+.2f} pp"
        )
