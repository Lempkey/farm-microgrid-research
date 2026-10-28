"""Summarise experiment 1 and apply its pre-registered decision rule (does capacity knowledge pay?)."""

import json
import statistics as st
import sys
from collections import defaultdict

YEARS = 5
PROBE = sys.argv[2] if len(sys.argv) > 2 else "reserve_probe"
rows = json.load(open(sys.argv[1]))
by = defaultdict(dict)
for r in rows:
    by[(r["farm"], r["pack"], r["partial"])].setdefault(r["policy"], {})[r["seed"]] = r


def mean(xs):
    return st.mean(xs) if xs else float("nan")


print(
    "farm  pack partial policy        unmetC/yr unmetNC/yr p95err rmse"
    "  detect(med,cens) cap_end probes/yr pairs/yr partial/yr"
)
verdicts = []
for key in sorted(by):
    pols = by[key]
    for pol in ("no_reserve", "reserve", "reserve_probe", "reserve_probe_stale", "reserve_probe_dawn", "oracle"):
        runs = list(pols[pol].values())
        det = [r["detect_days"] for r in runs if r["detect_days"] is not None]
        cens = sum(r["detect_days"] is None for r in runs) if key[1] == "used" else 0
        print(
            f"{key[0]:5} {key[1]:4} {key[2]:5}  {pol:19} {mean([r['unmet_critical_kwh'] for r in runs]) / YEARS:8.1f}"
            f" {mean([r['unmet_noncritical_kwh'] for r in runs]) / YEARS:9.1f}"
            f" {100 * mean([r['p95_abs_error'] for r in runs]):6.2f} {100 * mean([r['rmse_error'] for r in runs]):5.2f}"
            f"  {(st.median(det) if det else float('nan')):6.1f},{cens:2d}"
            f"  {100 * mean([r['final_capacity'] for r in runs]):6.1f}"
            f" {mean([r['probes'] for r in runs]) / YEARS:8.1f} {mean([r['pair_samples'] for r in runs]) / YEARS:8.1f}"
            f" {mean([r['partial_samples'] for r in runs]) / YEARS:9.1f}"
        )
    p, o = pols["reserve"], pols["oracle"]
    pr = pols[PROBE]
    seeds = sorted(p)

    def uc(d: dict, seeds: list = seeds) -> float:
        return sum(d[s]["unmet_critical_kwh"] for s in seeds)

    gap = uc(p) - uc(o)
    info_matters_unmet = uc(p) > 0 and gap / uc(p) >= 0.10
    det_p = [p[s]["detect_days"] for s in seeds if p[s]["detect_days"] is not None]
    det_pr = [pr[s]["detect_days"] for s in seeds if pr[s]["detect_days"] is not None]
    lat_p = st.median(det_p) if det_p else None
    lat_pr = st.median(det_pr) if det_pr else None
    info_matters_latency = key[1] == "used" and (lat_p is None or lat_p > 14)
    recovered = (uc(p) - uc(pr)) / gap if gap > 0 else float("nan")
    halves = lat_p is not None and lat_pr is not None and lat_pr <= lat_p / 2
    wear_pp = 100 * mean([p[s]["final_capacity"] - pr[s]["final_capacity"] for s in seeds])
    extra_unmet = (uc(pr) - uc(p)) / YEARS / len(seeds)
    passes = (
        (info_matters_unmet or info_matters_latency)
        and ((gap > 0 and recovered >= 0.25) or halves)
        and wear_pp <= 0.5
        and extra_unmet <= 0.0
    )
    verdicts.append((key, passes))
    print(
        f"   -> oracle gap {gap / YEARS / len(seeds):.1f} kWh/yr ({100 * gap / uc(p) if uc(p) else 0:.0f}% of passive);"
        f" latency passive {lat_p}, probe {lat_pr}; probes recover {recovered:.0%} of gap;"
        f" probe wear {wear_pp:.2f} pp; extra unmet critical {extra_unmet:+.1f} kWh/yr;"
        f" capacity-knowledge PASS={passes}"
    )
print("\nCases passing:", [k for k, v in verdicts if v] or "none")
