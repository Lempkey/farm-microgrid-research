"""Health-learning microgrid simulation. Synthetic sites keep each behaviour checkable by hand."""

import math

import pytest

from farmsim.coldroom import ColdRoom
from farmsim.healthsim import (
    Farm,
    Pack,
    Policy,
    Sensing,
    Site,
    cycle_wear,
    kalman_update,
    probe_pump_kwh,
    required_energy,
    rng_stream,
    simulate,
)

DAYS = 24


def sunny_site(days: int = 60, peak: float = 1.0) -> Site:
    pv = [max(0.0, peak * math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0 for h in range(DAYS)]
    return Site(pv_kw_per_kwp=tuple(pv * days), ambient_c=tuple([25.0] * DAYS * days))


def dark_site(days: int = 10) -> Site:
    return Site(pv_kw_per_kwp=tuple([0.0] * DAYS * days), ambient_c=tuple([25.0] * DAYS * days))


def small_farm(pv_kwp: float = 6.0, critical_kw: float = 0.3) -> Farm:
    return Farm(
        pv_kwp=pv_kwp,
        critical_kw=critical_kw,
        household_kw=tuple([0.1] * 18 + [0.5] * 5 + [0.1]),
        irrigation_kwh=tuple([4.0] * 12),
        tank_kwh=40.0,
    )


def test_full_cycle_wear_equals_full_cycle_fade():
    assert cycle_wear(soc_before=1.0, soc_after=0.0, fade_per_full_cycle=1e-4, exponent=1.7) == pytest.approx(1e-4)


def test_two_half_cycles_wear_less_than_one_full_cycle():
    half = cycle_wear(1.0, 0.5, 1e-4, 1.7)
    assert 2 * half < cycle_wear(1.0, 0.0, 1e-4, 1.7)


def test_kalman_update_moves_estimate_towards_measurement_and_shrinks_variance():
    est, var = kalman_update(est=1.0, var=0.03**2, z=0.9, r=0.01**2)
    assert 0.9 < est < 0.92
    assert var < 0.01**2


def test_energy_balance_closes_for_every_load_class():
    res = simulate(sunny_site(), small_farm(), Pack(nominal_kwh=15), Sensing(), Policy(), seed=1)
    assert res.served_kwh + res.unmet_critical_kwh + res.unmet_noncritical_kwh == pytest.approx(res.demand_kwh)


def test_dark_site_drains_pack_and_leaves_critical_load_unmet():
    res = simulate(dark_site(), small_farm(), Pack(nominal_kwh=10), Sensing(), Policy(reserve_soc=0.35), seed=1)
    assert res.unmet_critical_kwh > 0
    assert res.empty_anchors >= 1


def test_reserve_sheds_noncritical_load_before_critical_load_suffers():
    farm = small_farm(critical_kw=0.05)
    res = simulate(dark_site(days=2), farm, Pack(nominal_kwh=10), Sensing(), Policy(reserve_soc=0.35, oracle=True))
    assert res.unmet_noncritical_kwh > 0
    assert res.unmet_critical_kwh == 0


def test_frequent_full_empty_pairs_track_capacity_closely():
    pack = Pack(nominal_kwh=8, initial_capacity=0.9)
    sensing = Sensing(prior_capacity=1.0, prior_sigma=0.1)
    res = simulate(sunny_site(days=120), small_farm(pv_kwp=6), pack, sensing, Policy(reserve_soc=0.05), seed=3)
    assert res.pair_samples > 20
    assert res.final_abs_error < 0.02


def test_no_probes_without_probe_policy_and_some_with_it():
    pack = Pack(nominal_kwh=30, initial_capacity=0.85)
    sensing = Sensing(prior_sigma=0.08)
    passive = simulate(sunny_site(days=60), small_farm(pv_kwp=10), pack, sensing, Policy(), seed=2)
    probing = simulate(sunny_site(days=60), small_farm(pv_kwp=10), pack, sensing, Policy(probe=True), seed=2)
    assert passive.probes == 0
    assert probing.probes > 0
    assert probing.pair_samples > passive.pair_samples


def test_step_loss_is_detected_and_latency_reported():
    pack = Pack(nominal_kwh=8, step_hour=30 * DAYS, step_loss=0.15)
    res = simulate(sunny_site(days=90), small_farm(pv_kwp=6), pack, Sensing(), Policy(reserve_soc=0.05), seed=4)
    assert res.detect_days is not None
    assert res.detect_days < 15


def test_stale_probe_fires_when_no_pair_sample_for_too_long_even_if_filter_is_confident():
    pack = Pack(nominal_kwh=30)
    sensing = Sensing(prior_sigma=0.005)
    policy = Policy(probe=True, probe_trigger_sigma=1.0, probe_max_gap_days=14)
    res = simulate(sunny_site(days=60), small_farm(pv_kwp=10), pack, sensing, policy, seed=2)
    assert res.probes >= 2


def test_dawn_timed_probe_keeps_cold_room_served_while_still_reaching_empty():
    pack = Pack(nominal_kwh=30)
    sensing = Sensing(prior_sigma=0.005)
    policy = Policy(probe=True, probe_trigger_sigma=1.0, probe_max_gap_days=14, probe_dawn_timed=True)
    res = simulate(sunny_site(days=60), small_farm(pv_kwp=10), pack, sensing, policy, seed=2)
    assert res.probes >= 2
    assert res.empty_anchors >= 2
    assert res.unmet_critical_kwh == 0


def test_required_energy_covers_a_dark_horizon_including_discharge_losses():
    req = required_energy(pv_kw=[0.0] * 10, load_kw=[1.0] * 10, capacity_kwh=50, efficiency=0.95)
    assert req.initial_kwh == pytest.approx(10 / math.sqrt(0.95))
    assert req.feasible


def test_required_energy_only_counts_load_until_the_pack_can_refill():
    pv = [0.0] * 5 + [10.0] * 5 + [0.0] * 5
    req = required_energy(pv_kw=pv, load_kw=[1.0] * 15, capacity_kwh=50, efficiency=1.0)
    assert req.initial_kwh == pytest.approx(5.0)
    assert req.max_required_kwh == pytest.approx(5.0)


def test_required_energy_request_never_exceeds_capacity_but_flags_the_impossible_horizon():
    req = required_energy([0.0] * 100, [1.0] * 100, capacity_kwh=20, efficiency=1.0)
    assert req.initial_kwh == pytest.approx(20)
    assert not req.feasible
    assert req.max_required_kwh == pytest.approx(100)


def test_required_energy_flags_a_later_dark_spell_the_pack_cannot_hold_even_after_refilling():
    pv = [0.0] * 2 + [50.0] * 2 + [0.0] * 30  # refill, then 30 h at 1 kW against a 20 kWh pack
    req = required_energy(pv, [1.0] * 34, capacity_kwh=20, efficiency=1.0)
    assert req.initial_kwh == pytest.approx(2.0)  # only the first two hours need stored energy now
    assert not req.feasible
    assert req.max_required_kwh == pytest.approx(30)


def test_required_energy_is_feasible_when_need_exactly_fills_the_pack():
    req = required_energy([0.0] * 20, [1.0] * 20, capacity_kwh=20, efficiency=1.0)
    assert req.feasible
    assert req.initial_kwh == pytest.approx(20)


def cloudy_spell_site() -> Site:
    sunny = sunny_site(days=10)
    pv = list(sunny.pv_kw_per_kwp)
    for day in (5, 6):
        pv[day * DAYS : (day + 1) * DAYS] = [0.05 * x for x in pv[day * DAYS : (day + 1) * DAYS]]
    return Site(pv_kw_per_kwp=tuple(pv), ambient_c=sunny.ambient_c)


def test_perfect_forecast_reserve_protects_cold_room_better_than_fixed_reserve_through_a_cloudy_spell():
    farm = small_farm(pv_kwp=8, critical_kw=0.4)
    pack = Pack(nominal_kwh=20)
    fixed = simulate(cloudy_spell_site(), farm, pack, Sensing(), Policy(reserve_soc=0.35, oracle=True))
    perfect = Policy(forecast_reserve=True, forecast_sigmas=(0, 0, 0), oracle=True)
    smart = simulate(cloudy_spell_site(), farm, pack, Sensing(), perfect)
    assert smart.unmet_critical_kwh < fixed.unmet_critical_kwh
    assert smart.served_kwh + smart.unmet_critical_kwh + smart.unmet_noncritical_kwh == pytest.approx(smart.demand_kwh)


def test_noncritical_load_never_takes_the_estimate_below_the_reserve_within_an_hour():
    farm = small_farm(critical_kw=0.05)
    res = simulate(dark_site(days=1), farm, Pack(nominal_kwh=10), Sensing(), Policy(reserve_soc=0.35, oracle=True))
    served_nc = res.served_kwh - 0.05 * 24
    assert served_nc <= (1 - 0.35) * 10 * math.sqrt(0.95) + 1e-9


def cold_farm(room: ColdRoom, pv_kwp: float = 8.0) -> Farm:
    return Farm(
        pv_kwp=pv_kwp,
        critical_kw=0.0,
        household_kw=tuple([0.1] * 18 + [0.5] * 5 + [0.1]),
        irrigation_kwh=tuple([4.0] * 12),
        tank_kwh=40.0,
        cold_room=room,
    )


def test_cold_room_stays_cold_on_a_sunny_site_and_energy_balances():
    res = simulate(sunny_site(days=10), cold_farm(ColdRoom()), Pack(nominal_kwh=20), Sensing(), Policy(oracle=True))
    assert res.hours_above_max == 0
    assert res.served_kwh + res.unmet_critical_kwh + res.unmet_noncritical_kwh == pytest.approx(res.demand_kwh)


def test_cold_room_warms_past_the_limit_when_the_battery_runs_out():
    res = simulate(dark_site(days=4), cold_farm(ColdRoom()), Pack(nominal_kwh=5), Sensing(), Policy(oracle=True))
    assert res.hours_above_max > 0
    assert res.degree_hours_above_max > 0


def test_surplus_precooling_and_pumping_use_only_curtailed_pv():
    farm, pack = cold_farm(ColdRoom()), Pack(nominal_kwh=20)
    base = simulate(sunny_site(days=10), farm, pack, Sensing(), Policy(oracle=True))
    multi = simulate(
        sunny_site(days=10), farm, pack, Sensing(), Policy(oracle=True, precool_surplus=True, pump_surplus=True)
    )
    assert multi.precool_kwh > 0
    assert multi.spill_pumped_kwh > 0
    assert multi.final_capacity == pytest.approx(base.final_capacity, abs=1e-4)


def test_precooling_and_cold_credit_reduce_hot_hours_through_a_cloudy_spell():
    farm, pack = cold_farm(ColdRoom(), pv_kwp=6.0), Pack(nominal_kwh=12)
    need = dict(forecast_reserve=True, forecast_sigmas=(0, 0, 0), forecast_pessimism_z=0, oracle=True)
    base = simulate(cloudy_spell_site(), farm, pack, Sensing(), Policy(**need))
    multi = simulate(cloudy_spell_site(), farm, pack, Sensing(), Policy(**need, precool_surplus=True, cold_credit=True))
    assert multi.hours_above_max <= base.hours_above_max
    assert multi.unmet_critical_kwh < base.unmet_critical_kwh


def test_probe_pumping_counts_loads_once_when_pv_exceeds_them():
    pump = probe_pump_kwh(headroom_kwh=0.0, eta=1.0, pv=3.0, critical=0.5, nc_served=0.5, pump_kw=5.0, tank_room=10.0)
    assert pump == pytest.approx(2.0)


def test_probe_pumping_at_night_spends_battery_headroom_net_of_loads():
    pump = probe_pump_kwh(headroom_kwh=5.0, eta=0.9, pv=0.0, critical=0.5, nc_served=0.5, pump_kw=9.0, tank_room=10.0)
    assert pump == pytest.approx(5.0 * 0.9 - 1.0)


def test_probe_pumping_is_bounded_by_pump_power_and_tank_room():
    args = dict(headroom_kwh=50.0, eta=1.0, pv=0.0, critical=0.0, nc_served=0.0)
    assert probe_pump_kwh(**args, pump_kw=2.0, tank_room=10.0) == pytest.approx(2.0)
    assert probe_pump_kwh(**args, pump_kw=2.0, tank_room=0.5) == pytest.approx(0.5)
    assert probe_pump_kwh(headroom_kwh=0.0, eta=1.0, pv=0.0, critical=1.0, nc_served=0.0, pump_kw=2.0, tank_room=5) == 0


def test_random_streams_are_stable_and_distinct_per_purpose():
    assert rng_stream(3, 1).random() == rng_stream(3, 1).random()
    draws = {rng_stream(seed, k).random() for seed in range(5) for k in range(4)}
    assert len(draws) == 20


def test_probe_forecast_draws_do_not_change_measurement_noise():
    pack, sensing = Pack(nominal_kwh=8, initial_capacity=0.9), Sensing(prior_capacity=1.0, prior_sigma=0.1)
    site, farm = sunny_site(days=120), small_farm(pv_kwp=6)
    plain = simulate(site, farm, pack, sensing, Policy(reserve_soc=0.05), seed=3)
    never_probes = Policy(reserve_soc=0.05, probe=True, probe_trigger_sigma=0.0, probe_forecast_margin=1e9)
    drawing = simulate(site, farm, pack, sensing, never_probes, seed=3)  # draws a forecast every night
    assert drawing.probes == 0
    assert drawing.pair_samples == plain.pair_samples > 20
    assert drawing.final_abs_error == plain.final_abs_error
