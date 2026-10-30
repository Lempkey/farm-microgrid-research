"""Time 20 representative 5-year runs on one core, to ground compute estimates before a sweep.

Covers fixed reserves with learning, probing, the oracle, the need-sized reserve, and the cold room
with and without multi-store, on new and used packs. Prints the median and p95 seconds per run and what
a sweep of N runs takes at MAX_WORKERS processes.

Usage: python experiments/benchmark.py <power_json_dir>
"""

import statistics as st
import sys
import time
from dataclasses import replace
from pathlib import Path

from common import BATTERY_KWH, FARMS, MAX_WORKERS, load_site, pack_and_sensing, run_window

from farmsim.coldroom import ColdRoom
from farmsim.healthsim import Policy, simulate

NEED = dict(forecast_reserve=True, forecast_pessimism_z=0)
MULTI = dict(precool_surplus=True, pump_surplus=True, cold_credit=True)
CONFIGS = [
    ("fixed", False, Policy(reserve_soc=0.35)),
    ("probe", False, Policy(reserve_soc=0.35, probe=True, probe_max_gap_days=14)),
    ("oracle", False, Policy(reserve_soc=0.35, oracle=True)),
    ("need_sized", False, Policy(**NEED)),
    ("cold_multi", True, Policy(**NEED, **MULTI)),
]
SWEEPS = (1_440, 5_000, 20_000)


def main() -> None:
    site = load_site(Path(sys.argv[1]))
    times = []
    for i, (name, cold, policy) in enumerate(CONFIGS * 2):
        for kind in ("new", "used"):
            seed = i
            farm = replace(FARMS["base"], critical_kw=0.0, cold_room=ColdRoom()) if cold else FARMS["base"]
            pack, sensing = pack_and_sensing(kind, BATTERY_KWH["base"], seed, 0.04)
            start, hours = run_window(site, seed)
            t0 = time.perf_counter()
            simulate(site, farm, pack, sensing, policy, hours, start, seed)
            times.append(time.perf_counter() - t0)
            print(f"{name:11} {kind:4} seed {seed}: {times[-1]:.3f} s")
    times.sort()
    median, p95 = st.median(times), times[min(len(times) - 1, int(0.95 * len(times)))]
    print(f"\n{len(times)} runs: median {median:.3f} s, p95 {p95:.3f} s per 5-year run (one core)")
    for n in SWEEPS:
        at_median, at_p95 = (n * x / MAX_WORKERS / 60 for x in (median, p95))
        print(f"  {n:6d} runs at {MAX_WORKERS} workers: {at_median:.1f} min at median pace, {at_p95:.1f} min at p95")


if __name__ == "__main__":
    main()
