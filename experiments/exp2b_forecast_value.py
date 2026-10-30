"""Experiment 2b (post hoc, labelled as such): value of forecast information, frontier against frontier.

Perfect-weather and climatology-only reserves, each swept over safety margins, base sizing.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import BATTERY_KWH, FARMS, MAX_WORKERS, YEARS, load_site, pack_and_sensing, run_window

from farmsim.healthsim import Policy, Site, simulate

MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25)
POLICIES = {
    **{
        f"perfect_m{int(m * 100):02d}": Policy(
            forecast_reserve=True, forecast_sigmas=(0.0, 0.0, 0.0), forecast_pessimism_z=0, reserve_margin=m
        )
        for m in MARGINS
    },
    **{
        f"clim_m{int(m * 100):02d}": Policy(
            forecast_reserve=True, forecast_climatology=True, forecast_pessimism_z=0, reserve_margin=m
        )
        for m in MARGINS
    },
}
_SITE: Site | None = None


def _init(site: Site) -> None:
    global _SITE
    _SITE = site


def _worker(args: tuple) -> dict:
    pack_kind, policy_key, seed = args
    pack, sensing = pack_and_sensing(pack_kind, BATTERY_KWH["base"], seed, 0.04)
    start, hours = run_window(_SITE, seed)
    res = simulate(_SITE, FARMS["base"], pack, sensing, POLICIES[policy_key], hours, start, seed)
    return dict(farm="base", pack=pack_kind, policy=policy_key, seed=seed, **res.__dict__)


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    site = load_site(folder)
    jobs = [(k, p, s) for k in ("new", "used") for p in POLICIES for s in range(20)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init, initargs=(site,)) as ex:
        rows = list(ex.map(_worker, jobs, chunksize=4))
    out.write_text(json.dumps(rows))
    for k in ("new", "used"):
        print(f"== {k} pack: policy, unmet critical kWh/yr, unmet non-critical kWh/yr")
        for p in POLICIES:
            rs = [r for r in rows if r["pack"] == k and r["policy"] == p]
            c = sum(r["unmet_critical_kwh"] for r in rs) / len(rs) / YEARS
            nc = sum(r["unmet_noncritical_kwh"] for r in rs) / len(rs) / YEARS
            print(f"  {p:12} {c:7.2f} {nc:8.1f}")


if __name__ == "__main__":
    main()
