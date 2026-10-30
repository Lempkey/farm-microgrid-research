"""Shared setup for the experiments: weather loading, placeholder farm sizing and battery packs.

All farm, pack and sensing numbers are placeholders, not measurements of a real site.
"""

import json
import os
from pathlib import Path

from farmsim.healthsim import STREAM_SETUP, Farm, Pack, Sensing, Site, rng_stream

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))  # keep sweeps from taking every core

PERFORMANCE_RATIO = 0.78  # estimate: horizontal GHI to AC output, temperature losses included
YEARS = 5
H = 8760

HOUSEHOLD = tuple([0.1] * 6 + [0.3] * 3 + [0.2] * 9 + [0.6] * 4 + [0.3] * 2)  # 6.3 kWh/day
IRRIGATION = (8, 8, 8, 8, 4, 1.5, 1.5, 1.5, 1.5, 1.5, 4, 8)  # kWh/day by month; dry season Dec-Apr
FARMS = {
    "base": Farm(pv_kwp=7.0, critical_kw=0.45, household_kw=HOUSEHOLD, irrigation_kwh=IRRIGATION),
    "tight": Farm(pv_kwp=5.5, critical_kw=0.45, household_kw=HOUSEHOLD, irrigation_kwh=IRRIGATION),
}
BATTERY_KWH = {"base": 25.0, "tight": 16.0}


def load_site(folder: Path) -> Site:
    pv, temp = [], []
    for year in range(2015, 2025):
        p = json.loads((folder / f"p{year}.json").read_text())["properties"]["parameter"]
        for key in sorted(p["ALLSKY_SFC_SW_DWN"]):
            g, t = p["ALLSKY_SFC_SW_DWN"][key], p["T2M"][key]
            pv.append(max(g, 0.0) / 1000 * PERFORMANCE_RATIO)
            temp.append(t if t > -100 else 26.0)
    return Site(pv_kw_per_kwp=tuple(pv), ambient_c=tuple(temp))


def pack_and_sensing(kind: str, battery_kwh: float, seed: int, partial: float | None) -> tuple[Pack, Sensing]:
    rng = rng_stream(seed, STREAM_SETUP)
    gain = rng.gauss(0.0, 0.005)
    if kind == "new":
        pack = Pack(nominal_kwh=battery_kwh)
        sensing = Sensing(partial_sigma=partial, gain_error=gain)
    else:  # used pack in heat, seller's claim uncertain, one module lost at a random time in years 1-4
        pack = Pack(
            nominal_kwh=battery_kwh,
            initial_capacity=0.82,
            calendar_fade_per_year=0.025,
            cycle_fade_per_full_cycle=0.2 / 3000,
            step_hour=rng.randrange(H, 4 * H),
            step_loss=0.15,
        )
        sensing = Sensing(
            partial_sigma=partial, gain_error=gain, prior_capacity=0.90, prior_sigma=0.08, drift_per_day=0.05 / 365
        )
    return pack, sensing
