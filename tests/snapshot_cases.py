"""Regression cases: six configurations on a synthetic, variable-weather site.

The site is generated from a fixed seed, so the cases need no downloaded data. Outputs are compared
with `snapshots.json` at a tolerance. An intended behaviour change regenerates the file and comes with
a reviewed delta table:

    UPDATE_SNAPSHOTS=1 uv run pytest tests/test_snapshots.py
    uv run python experiments/snapshot_delta.py <old.json> tests/snapshots.json
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict

from farmsim.coldroom import ColdRoom
from farmsim.healthsim import Farm, Pack, Policy, Sensing, Site, simulate

DAYS = 150
WEATHER_SEED = 7


def variable_site(days: int = DAYS) -> Site:
    """Clear-sky arc scaled by a random daily clearness in [0.3, 1.2]; warm, with a daily swing."""
    rng = random.Random(WEATHER_SEED)
    pv, ambient = [], []
    for _ in range(days):
        clearness = rng.uniform(0.3, 1.2)
        for h in range(24):
            arc = math.sin(math.pi * (h - 6) / 12) if 6 <= h <= 18 else 0.0
            pv.append(0.85 * clearness * max(0.0, arc))
            ambient.append(27.0 + 4.0 * math.sin(math.pi * (h - 9) / 12))
    return Site(pv_kw_per_kwp=tuple(pv), ambient_c=tuple(ambient))


HOUSEHOLD = tuple([0.1] * 6 + [0.3] * 3 + [0.2] * 9 + [0.6] * 4 + [0.3] * 2)


def _farm(room: ColdRoom | None = None, pv_kwp: float = 7.0) -> Farm:
    return Farm(
        pv_kwp=pv_kwp,
        critical_kw=0.0 if room else 0.45,
        household_kw=HOUSEHOLD,
        irrigation_kwh=tuple([4.0] * 12),
        cold_room=room,
    )


def _used() -> tuple[Pack, Sensing]:
    pack = Pack(nominal_kwh=16, initial_capacity=0.82, calendar_fade_per_year=0.025, step_hour=50 * 24, step_loss=0.15)
    return pack, Sensing(partial_sigma=0.04, gain_error=0.003, prior_capacity=0.9, prior_sigma=0.08)


CASES = {
    "fixed_new_partial": lambda: (_farm(), Pack(nominal_kwh=16), Sensing(partial_sigma=0.04), Policy()),
    "fixed_used_step": lambda: (_farm(), *_used(), Policy()),
    "probe_used_step": lambda: (_farm(pv_kwp=10.0), *_used(), Policy(probe=True, probe_max_gap_days=7)),
    "oracle_used_step": lambda: (_farm(), *_used(), Policy(oracle=True)),
    "need_sized_cold": lambda: (_farm(ColdRoom()), Pack(nominal_kwh=12), Sensing(), Policy(forecast_reserve=True)),
    "multi_store_bank": lambda: (
        _farm(ColdRoom(bank_kwh=6.0)),
        Pack(nominal_kwh=12),
        Sensing(),
        Policy(forecast_reserve=True, precool_surplus=True, pump_surplus=True, cold_credit=True),
    ),
}


def run_case(name: str) -> dict:
    farm, pack, sensing, policy = CASES[name]()
    return asdict(simulate(variable_site(), farm, pack, sensing, policy, seed=11))


def run_all() -> dict:
    return {name: run_case(name) for name in CASES}
