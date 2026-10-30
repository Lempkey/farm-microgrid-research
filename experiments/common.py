"""Shared setup for the experiments: weather loading, placeholder farm sizing and battery packs.

All farm, pack and sensing numbers are placeholders, not measurements of a real site.
"""

import json
import os
from pathlib import Path

from farmsim.healthsim import STREAM_SETUP, Farm, Pack, Sensing, Site, months_from_timestamps, rng_stream

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))  # keep sweeps from taking every core

PERFORMANCE_RATIO = 0.78  # estimate: horizontal GHI to AC output, temperature losses included
YEARS = 5
H = 8760
BASE_YEARS = range(2015, 2025)  # the weather behind the R-43 to R-45 headlines; seeds start at seed % 10
ALL_YEARS = range(2001, 2025)  # NASA POWER hourly coverage, for per-year statistics

HOUSEHOLD = tuple([0.1] * 6 + [0.3] * 3 + [0.2] * 9 + [0.6] * 4 + [0.3] * 2)  # 6.3 kWh/day
IRRIGATION = (8, 8, 8, 8, 4, 1.5, 1.5, 1.5, 1.5, 1.5, 4, 8)  # kWh/day by month; dry season Dec-Apr
FARMS = {
    "base": Farm(pv_kwp=7.0, critical_kw=0.45, household_kw=HOUSEHOLD, irrigation_kwh=IRRIGATION),
    "tight": Farm(pv_kwp=5.5, critical_kw=0.45, household_kw=HOUSEHOLD, irrigation_kwh=IRRIGATION),
}
BATTERY_KWH = {"base": 25.0, "tight": 16.0}


def load_site(folder: Path, years: range = BASE_YEARS) -> Site:
    pv, temp, keys = [], [], []
    for year in years:
        p = json.loads((folder / f"p{year}.json").read_text())["properties"]["parameter"]
        for key in sorted(p["ALLSKY_SFC_SW_DWN"]):
            g, t = p["ALLSKY_SFC_SW_DWN"][key], p["T2M"][key]
            pv.append(max(g, 0.0) / 1000 * PERFORMANCE_RATIO)
            temp.append(t if t > -100 else 26.0)
            keys.append(key)
    return Site(pv_kw_per_kwp=tuple(pv), ambient_c=tuple(temp), month_of_hour=months_from_timestamps(keys))


def year_starts(site: Site) -> list[int]:
    """Hour index of each 1 January in the site's calendar."""
    months = site.month_of_hour
    return [i for i in range(len(months)) if months[i] == 0 and (i == 0 or months[i - 1] == 11)]


def run_window(site: Site, seed: int, years: int = YEARS) -> tuple[int, int]:
    """(start hour, hours) of a run of whole calendar years starting at year `seed % n_years`, wrapping."""
    starts = year_starts(site)
    n = len(starts)
    ends = starts[1:] + [len(site.pv_kw_per_kwp)]
    first = seed % n
    return starts[first], sum(ends[(first + k) % n] - starts[(first + k) % n] for k in range(years))


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
