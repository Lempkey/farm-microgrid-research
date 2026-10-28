"""Experiment 2: a reserve sized from a weather forecast against the fixed-reserve frontier.

Usage: python experiments/exp2_need_sized_reserve.py <power_json_dir> [seeds] [out]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import BATTERY_KWH, FARMS, MAX_WORKERS, YEARS, H, load_site, pack_and_sensing

from farmsim.healthsim import Policy, Site, simulate

POLICIES = {
    **{f"fixed_{int(r * 100):02d}": Policy(reserve_soc=r) for r in (0.05, 0.20, 0.35, 0.50, 0.65, 0.80)},
    **{f"fc_z{z}": Policy(forecast_reserve=True, forecast_pessimism_z=z) for z in (0, 1, 2)},
    "fc_climatology": Policy(forecast_reserve=True, forecast_climatology=True),
    "fc_perfect": Policy(forecast_reserve=True, forecast_sigmas=(0.0, 0.0, 0.0), forecast_pessimism_z=0),
}

_SITE: Site | None = None


def _init(site: Site) -> None:
    global _SITE
    _SITE = site


def _worker(args: tuple) -> dict:
    farm_key, pack_kind, policy_key, seed = args
    pack, sensing = pack_and_sensing(pack_kind, BATTERY_KWH[farm_key], seed, 0.04)
    start = (seed % 10) * H
    res = simulate(_SITE, FARMS[farm_key], pack, sensing, POLICIES[policy_key], YEARS * H, start, seed)
    return dict(farm=farm_key, pack=pack_kind, policy=policy_key, seed=seed, **res.__dict__)


def main() -> None:
    folder = Path(sys.argv[1])
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else folder / "exp2_rows.json"
    site = load_site(folder)
    jobs = [(f, k, p, s) for f in FARMS for k in ("new", "used") for p in POLICIES for s in range(seeds)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init, initargs=(site,)) as ex:
        rows = list(ex.map(_worker, jobs, chunksize=4))
    out.write_text(json.dumps(rows))
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
