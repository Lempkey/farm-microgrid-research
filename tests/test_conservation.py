"""Energy conservation for every store, checked on hand-built and variable-weather cases."""

import pytest
from snapshot_cases import CASES, run_case
from test_healthsim import cold_farm, dark_site, small_farm, sunny_site

from farmsim.coldroom import ColdRoom
from farmsim.healthsim import Pack, Policy, Result, Sensing, simulate

ETA = 0.95**0.5


def _bus_balances(r: Result) -> None:
    """Sources (PV, battery) equal sinks (loads served, pumping, pre-cooling, charging, spill)."""
    sources = r.pv_kwh + r.battery_out_kwh
    sinks = r.served_kwh + r.pumped_kwh + r.precool_kwh + r.battery_in_kwh + r.spilled_kwh
    assert sources == pytest.approx(sinks, rel=1e-9, abs=1e-9)


def _battery_balances(r: Result, eta: float = ETA) -> None:
    expected = r.initial_energy_kwh + eta * r.battery_in_kwh - r.battery_out_kwh / eta - r.fade_clipped_kwh
    assert r.final_energy_kwh == pytest.approx(expected, rel=1e-9, abs=1e-9)


def _tank_balances(r: Result) -> None:
    assert r.final_tank_kwh == pytest.approx(r.pumped_kwh - r.tank_drawn_kwh, abs=1e-9)
    assert r.final_tank_kwh >= -1e-12


def _cold_balances(r: Result, cop: float) -> None:
    expected = r.initial_cold_kwh_th + cop * r.compressor_kwh - r.cold_leak_kwh_th
    assert r.final_cold_kwh_th == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_dark_day_drains_exactly_what_the_load_took_through_the_inverter():
    farm = small_farm(critical_kw=0.3)
    r = simulate(dark_site(days=1), farm, Pack(nominal_kwh=10, calendar_fade_per_year=0.0), Sensing(), Policy())
    assert r.pv_kwh == 0
    assert r.battery_out_kwh == pytest.approx(r.served_kwh)
    assert r.initial_energy_kwh - r.final_energy_kwh == pytest.approx(r.served_kwh / ETA)


@pytest.mark.parametrize(
    "case",
    [
        lambda: simulate(sunny_site(), small_farm(), Pack(nominal_kwh=15), Sensing(), Policy(), seed=1),
        lambda: simulate(dark_site(), small_farm(), Pack(nominal_kwh=10), Sensing(), Policy(), seed=1),
        lambda: simulate(
            sunny_site(days=60), small_farm(pv_kwp=10), Pack(nominal_kwh=30, initial_capacity=0.85),
            Sensing(prior_sigma=0.08), Policy(probe=True), seed=2,
        ),
    ],
)  # fmt: skip
def test_bus_battery_and_tank_balance_on_hand_built_sites(case):
    r = case()
    _bus_balances(r)
    _battery_balances(r)
    _tank_balances(r)


def test_cold_room_balances_when_it_warms_past_the_limit():
    room = ColdRoom()
    r = simulate(dark_site(days=4), cold_farm(room), Pack(nominal_kwh=5), Sensing(), Policy(oracle=True))
    assert r.final_cold_kwh_th < 0  # room above t_max, the q < 0 branch
    _cold_balances(r, room.cop)
    _bus_balances(r)
    _battery_balances(r)


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_store_balances_on_the_regression_cases(name):
    r = Result(**run_case(name))
    _bus_balances(r)
    _battery_balances(r)
    _tank_balances(r)
    farm = CASES[name]()[0]
    if farm.cold_room:
        _cold_balances(r, farm.cold_room.cop)


def test_capacity_fade_clips_stored_energy_and_accounts_for_it():
    pack = Pack(nominal_kwh=10, step_hour=5, step_loss=0.3)
    r = simulate(dark_site(days=1), small_farm(critical_kw=0.0), pack, Sensing(), Policy(reserve_soc=1.0))
    assert r.fade_clipped_kwh == pytest.approx(3.0, rel=1e-3)
    _battery_balances(r)
