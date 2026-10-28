"""Experiment 1: is knowing the battery's capacity worth much for dispatch?

Compares fixed reserves with and without capacity learning, probe nights, and an oracle that knows the
true capacity. Weather: NASA POWER hourly, Valencia area, Venezuela (10.16 N, 68.00 W).

Usage: python experiments/exp1_capacity_knowledge.py <power_json_dir> [seeds] [out]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import BATTERY_KWH, FARMS, MAX_WORKERS, YEARS, H, load_site, pack_and_sensing

from farmsim.healthsim import Policy, Site, simulate

POLICIES = {
    "no_reserve": Policy(reserve_soc=0.05),
    "reserve": Policy(reserve_soc=0.35),
    "reserve_probe": Policy(reserve_soc=0.35, probe=True),
    "reserve_probe_stale": Policy(reserve_soc=0.35, probe=True, probe_max_gap_days=14),
    "reserve_probe_dawn": Policy(reserve_soc=0.35, probe=True, probe_max_gap_days=14, probe_dawn_timed=True),
    "oracle": Policy(reserve_soc=0.35, oracle=True),
}
PARTIAL = {"2pp": 0.02, "4pp": 0.04, "8pp": 0.08, "none": None}


def run_one(args: tuple) -> dict:
    site, farm_key, pack_kind, partial_key, policy_key, seed = args
    pack, sensing = pack_and_sensing(pack_kind, BATTERY_KWH[farm_key], seed, PARTIAL[partial_key])
    start = (seed % 10) * H
    res = simulate(site, FARMS[farm_key], pack, sensing, POLICIES[policy_key], YEARS * H, start, seed)
    return dict(farm=farm_key, pack=pack_kind, partial=partial_key, policy=policy_key, seed=seed, **res.__dict__)


_SITE: Site | None = None


def _init(site: Site) -> None:
    global _SITE
    _SITE = site


def _worker(args: tuple) -> dict:
    return run_one((_SITE, *args))


def main() -> None:
    folder, seeds = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 20
    site = load_site(folder)
    mean_pv = sum(site.pv_kw_per_kwp) / len(site.pv_kw_per_kwp) * 24
    print(f"site: {len(site.pv_kw_per_kwp)} h; mean PV {mean_pv:.2f} kWh/kWp/day")
    jobs = [
        (f, k, p, pol, s)
        for f in FARMS
        for k in ("new", "used")
        for p in PARTIAL
        for pol in POLICIES
        for s in range(seeds)
        if f == "base" or p in ("4pp", "none")
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init, initargs=(site,)) as ex:
        rows = list(ex.map(_worker, jobs, chunksize=4))
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else folder / "exp1_rows.json"
    out.write_text(json.dumps(rows))
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
