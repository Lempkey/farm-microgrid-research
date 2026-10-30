"""Experiment 3: multi-store (cold room + water tank + battery) against battery-only, frontier against frontier.

Usage: python experiments/exp3_multi_store.py <power_json_dir> [seeds] [out]
Base sizing only (the pre-registered rule's scope). Cold room, bank and farm numbers are placeholders.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from common import BATTERY_KWH, FARMS, MAX_WORKERS, load_site, pack_and_sensing, run_window

from farmsim.coldroom import ColdRoom
from farmsim.healthsim import Policy, Site, simulate

MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25)
BANK_KWH_TH = 6.0
ROOMS = {"produce": ColdRoom(), "bank": ColdRoom(bank_kwh=BANK_KWH_TH)}
FAMILIES = {
    "batt_only": {},
    "multi": dict(precool_surplus=True, pump_surplus=True, cold_credit=True),
    "cold_only": dict(precool_surplus=True, cold_credit=True),
    "water_only": dict(pump_surplus=True),
}
EXTRA_BATTERY_KWH = BANK_KWH_TH / ColdRoom().cop  # the bank's electrical equivalent


def policy(family: str, margin: float) -> Policy:
    return Policy(forecast_reserve=True, forecast_pessimism_z=0, reserve_margin=margin, **FAMILIES[family])


_SITE: Site | None = None


def _init(site: Site) -> None:
    global _SITE
    _SITE = site


def _worker(args: tuple) -> dict:
    pack_kind, room_key, family, margin, seed = args
    big = room_key == "bigbatt"
    farm = replace(FARMS["base"], critical_kw=0.0, cold_room=ROOMS["produce" if big else room_key])
    battery = BATTERY_KWH["base"] + (EXTRA_BATTERY_KWH if big else 0.0)
    pack, sensing = pack_and_sensing(pack_kind, battery, seed, 0.04)
    start, hours = run_window(_SITE, seed)
    res = simulate(_SITE, farm, pack, sensing, policy(family, margin), hours, start, seed)
    return dict(pack=pack_kind, room=room_key, family=family, margin=margin, seed=seed, **res.__dict__)


def main() -> None:
    folder = Path(sys.argv[1])
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else folder / "exp3_rows.json"
    site = load_site(folder)
    cases = [(r, f) for r in ROOMS for f in FAMILIES] + [("bigbatt", "batt_only")]
    jobs = [
        (pack, room, fam, m, s)
        for pack in ("new", "used")
        for room, fam in cases
        for m in MARGINS
        for s in range(seeds)
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init, initargs=(site,)) as ex:
        rows = list(ex.map(_worker, jobs, chunksize=8))
    out.write_text(json.dumps(rows))
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
