"""Experiments 2, 2b and 3 re-run per weather year (2001-2024) and in four 5-year windows.

Each calendar year is a separate 1-year run from 1 January, with a full pack and the controller's prior;
seeds vary the forecast noise and pack draws and are averaged within the year at analysis. In annual runs
the used pack is an *initially degraded pack*: its module loss is switched off explicitly (in 5-year runs
it falls in years 2-4). The four non-overlapping 5-year windows (2001-05, 2006-10, 2011-15, 2016-20)
carry wear over 5 years and the module loss. Years are separate experimental units but consecutive
observations of one climate record, not independent draws. Policies,
packs and margins are exactly those of the original experiments.

Usage: python experiments/per_year.py <power_json_dir> <out.json> [seeds]
Analyse with experiments/summarise_per_year.py.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from common import ALL_YEARS, BATTERY_KWH, FARMS, MAX_WORKERS, load_site, pack_and_sensing, year_starts
from exp2_need_sized_reserve import POLICIES as EXP2_POLICIES
from exp2b_forecast_value import POLICIES as EXP2B_POLICIES
from exp3_multi_store import EXTRA_BATTERY_KWH, FAMILIES, MARGINS, ROOMS
from exp3_multi_store import policy as exp3_policy

from farmsim.healthsim import Site, simulate

SEEDS = 10
WINDOW_FIRST_YEARS = (2001, 2006, 2011, 2016)
WINDOW_YEARS = 5
PARTIAL_SIGMA = 0.04
KEEP = (
    "unmet_critical_kwh",
    "unmet_noncritical_kwh",
    "hours_above_max",
    "degree_hours_above_max",
    "final_capacity",
    "infeasible_reserve_hours",
)


def configs() -> list[dict]:
    """Every (experiment, sizing, pack, policy) of experiments 2, 2b and 3."""
    out = []
    for kind in ("new", "used"):
        for farm in FARMS:
            out += [dict(exp="2", farm=farm, pack=kind, policy=p) for p in EXP2_POLICIES]
        out += [dict(exp="2b", farm="base", pack=kind, policy=p) for p in EXP2B_POLICIES]
        cases = [(room, fam) for room in ROOMS for fam in FAMILIES] + [("bigbatt", "batt_only")]
        out += [
            dict(exp="3", farm="base", pack=kind, room=room, family=fam, margin=m, policy=f"{room}/{fam}/m{m:.2f}")
            for room, fam in cases
            for m in MARGINS
        ]
    return out


def units(site: Site) -> list[dict]:
    """One unit per calendar year, plus the four 5-year windows: (start hour, hours)."""
    starts = year_starts(site)
    ends = starts[1:] + [len(site.pv_kw_per_kwp)]
    years = list(ALL_YEARS)
    out = [
        dict(unit="year", year=y, start=starts[i], hours=ends[i] - starts[i], n_years=1) for i, y in enumerate(years)
    ]
    for first in WINDOW_FIRST_YEARS:
        i = years.index(first)
        span = ends[i + WINDOW_YEARS - 1] - starts[i]
        out.append(dict(unit="window", year=first, start=starts[i], hours=span, n_years=WINDOW_YEARS))
    return out


def run(site: Site, cfg: dict, unit: dict, seed: int) -> dict:
    run_seed = (unit["year"] - 2000) * 1000 + (500 if unit["unit"] == "window" else 0) + seed
    if cfg["exp"] == "3":
        big = cfg["room"] == "bigbatt"
        farm = replace(FARMS["base"], critical_kw=0.0, cold_room=ROOMS["produce" if big else cfg["room"]])
        battery = BATTERY_KWH["base"] + (EXTRA_BATTERY_KWH if big else 0.0)
        policy = exp3_policy(cfg["family"], cfg["margin"])
    else:
        farm, battery = FARMS[cfg["farm"]], BATTERY_KWH[cfg["farm"]]
        policy = (EXP2_POLICIES if cfg["exp"] == "2" else EXP2B_POLICIES)[cfg["policy"]]
    pack, sensing = pack_and_sensing(cfg["pack"], battery, run_seed, PARTIAL_SIGMA)
    if unit["unit"] == "year":  # an annual run is an initially degraded pack: no module loss, by design
        pack = replace(pack, step_hour=None, step_loss=0.0)
    res = simulate(site, farm, pack, sensing, policy, unit["hours"], unit["start"], run_seed)
    kept = {k: getattr(res, k) for k in KEEP}
    return {**cfg, "unit": unit["unit"], "year": unit["year"], "n_years": unit["n_years"], "seed": seed, **kept}


_SITE: Site | None = None


def _init(site: Site) -> None:
    global _SITE
    _SITE = site


def _worker(job: tuple) -> dict:
    return run(_SITE, *job)


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else SEEDS
    site = load_site(folder, ALL_YEARS)
    jobs = [(c, u, s) for c in configs() for u in units(site) for s in range(seeds)]
    print(f"{len(jobs)} runs on {MAX_WORKERS} workers")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init, initargs=(site,)) as ex:
        rows = list(ex.map(_worker, jobs, chunksize=16))
    out.write_text(json.dumps(rows))
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
