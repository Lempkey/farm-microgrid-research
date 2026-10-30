"""Hourly off-grid farm microgrid with a battery whose capacity must be learned (experiment 1).

Truth: an LFP pack fades with temperature-scaled calendar ageing and depth-weighted cycling, and may
lose a module abruptly (`step_hour`). The BMS cuts off at `floor_soc` of the *true* capacity.

Controller: counts state of charge against its own capacity estimate, sheds non-critical load below
`reserve_soc`, and (if `probe`) runs a probe night: pumps water into the irrigation tank until the
pack reaches the informative low end, so the next full charge yields a capacity sample.

Learning: a scalar Kalman filter on capacity (fraction of nominal), fed by full-empty pairs (noise
calibrated to field data; see field_capacity/) and optionally by partial-charge estimates of noise `partial_sigma`.
A sample more than 4 sigma off is held, not used; a second consistent one within 30 days confirms a step and
resets the filter (`filter_update`). The estimate stays within physical bounds.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from farmsim.coldroom import ColdRoom, compressor_need, leak_kwh, planned_loads, temperature

HOURS_PER_DAY = 24
HOURS_PER_YEAR = 8760
FULL_SOC = 0.995
PARTIAL_START_SOC = 0.40  # a charge from below this crosses the LFP features partial methods need
REARM_SOC = 0.20  # an empty anchor re-arms once the pack climbs back above this
DETECT_TOLERANCE = 0.03
JUMP_SIGMAS = 4.0  # a sample further off than this is held as a step candidate
CONFIRM_SIGMAS = 3.0  # a second outlier this close to the first confirms the step
CANDIDATE_LIFETIME_DAYS = 30
JUMP_RESET_SIGMA = 0.015  # variance floor after a confirmed step: two samples may share a bias
CAPACITY_BOUNDS = (0.30, 1.05)  # fraction of nominal; above 1 allows for an uncertain nameplate
NIGHT_START, NIGHT_END = 18, 6
PV_COVERS_LOAD_HOUR = 7  # dawn-timed probes keep enough for the cold room until PV takes over
PROBE_SOC_MARGIN = 0.02
DAWN_LANDING_BELOW_EMPTY = 0.02
FORECAST_HORIZON_H = 72
MAX_RESERVE_SOC = 0.95


@dataclass(frozen=True)
class Site:
    pv_kw_per_kwp: tuple[float, ...]  # hourly output per kWp, system losses included
    ambient_c: tuple[float, ...]


@dataclass(frozen=True)
class Farm:
    pv_kwp: float
    critical_kw: float  # cold room, 24 h
    household_kw: tuple[float, ...]  # 24 hourly values
    irrigation_kwh: tuple[float, ...]  # daily pumping energy by calendar month (12 values)
    irrigation_hours: tuple[int, ...] = (9, 10, 11, 12, 13)
    pump_kw: float = 2.0
    tank_kwh: float = 16.0  # pumping energy the tank can hold
    cold_room: ColdRoom | None = None  # if set, the critical load is this room's compressor, not critical_kw


@dataclass(frozen=True)
class Pack:
    nominal_kwh: float
    initial_capacity: float = 1.0  # fraction of nominal
    calendar_fade_per_year: float = 0.015  # at 25 C, doubling per 10 C
    cycle_fade_per_full_cycle: float = 0.2 / 6000
    depth_exponent: float = 1.7
    floor_soc: float = 0.05
    round_trip_efficiency: float = 0.95
    temp_offset_c: float = 3.0
    step_hour: int | None = None
    step_loss: float = 0.0  # fraction of nominal lost at step_hour


@dataclass(frozen=True)
class Sensing:
    pair_sigma: float = 0.01  # full-empty pair spanning 0.9 of SoC (field data: ~1 pp)
    partial_sigma: float | None = None  # None: no partial-charge estimator
    gain_error: float = 0.0
    empty_soc: float = 0.10
    prior_capacity: float = 1.0
    prior_sigma: float = 0.03
    drift_per_day: float = 0.02 / 365
    process_sigma_per_day: float = 0.0005


@dataclass(frozen=True)
class Policy:
    reserve_soc: float = 0.35
    probe: bool = False
    probe_trigger_sigma: float = 0.02
    probe_forecast_margin: float = 1.3
    probe_min_gap_days: int = 7
    probe_max_gap_days: int | None = None  # also probe when no pair sample for this long (catches silent steps)
    probe_dawn_timed: bool = False  # drain only what the cold room will not need before PV takes over
    forecast_sigma: float = 0.3  # lognormal error on next-day PV energy
    oracle: bool = False  # controller uses the true capacity
    forecast_reserve: bool = False  # keep what the cold room needs over the forecast horizon instead of reserve_soc
    forecast_sigmas: tuple[float, ...] = (0.25, 0.35, 0.45)  # daily PV error by lead day (lognormal)
    forecast_pessimism_z: float = 1.0
    forecast_climatology: bool = False  # no forecast link: plan on the monthly-hourly mean
    reserve_margin: float = 0.02
    precool_surplus: bool = False  # PV that would be curtailed pre-cools the cold room towards t_min
    pump_surplus: bool = False  # ... and pumps water into the irrigation tank
    cold_credit: bool = False  # the need-sized reserve counts cold stored above the setpoint


@dataclass(frozen=True)
class Result:
    demand_kwh: float
    served_kwh: float
    unmet_critical_kwh: float
    unmet_noncritical_kwh: float
    pumped_kwh: float
    final_capacity: float
    final_abs_error: float
    rmse_error: float
    p95_abs_error: float
    detect_days: float | None
    pair_samples: int
    partial_samples: int
    empty_anchors: int
    probes: int
    hours_above_max: int = 0
    degree_hours_above_max: float = 0.0
    precool_kwh: float = 0.0
    spill_pumped_kwh: float = 0.0
    # energy accounting, for conservation checks: AC/DC side, battery, cold room, tank
    pv_kwh: float = 0.0
    battery_in_kwh: float = 0.0  # drawn from the bus to charge, before losses
    battery_out_kwh: float = 0.0  # delivered to the bus, after losses
    spilled_kwh: float = 0.0  # PV nothing could use
    fade_clipped_kwh: float = 0.0  # stored energy lost when capacity fell below it
    initial_energy_kwh: float = 0.0
    final_energy_kwh: float = 0.0
    compressor_kwh: float = 0.0  # electric, cooling plus pre-cooling
    cold_leak_kwh_th: float = 0.0
    initial_cold_kwh_th: float = 0.0
    final_cold_kwh_th: float = 0.0
    tank_drawn_kwh: float = 0.0
    final_tank_kwh: float = 0.0
    infeasible_horizons: int = 0  # hours whose need-sized reserve came from an impossible horizon
    held_outliers: int = 0  # capacity samples held as step candidates
    confirmed_jumps: int = 0
    estimate_clips: int = 0  # times the capacity estimate hit CAPACITY_BOUNDS


def cycle_wear(soc_before: float, soc_after: float, fade_per_full_cycle: float, exponent: float) -> float:
    """Depth-weighted wear of a discharge step: a full 1 -> 0 discharge costs one full cycle."""
    depth_before, depth_after = 1.0 - soc_before, 1.0 - soc_after
    return fade_per_full_cycle * (max(depth_after, 0.0) ** exponent - max(depth_before, 0.0) ** exponent)


@dataclass(frozen=True)
class RequiredEnergy:
    initial_kwh: float  # implementable reserve request now, never above capacity
    feasible: bool  # False: the horizon's load cannot be met even from a suitably charged pack
    max_required_kwh: float  # largest stored energy the horizon needs at any hour, if capacity were unlimited


def required_energy(pv_kw: list[float], load_kw: list[float], capacity_kwh: float, efficiency: float) -> RequiredEnergy:
    """Backward recursion over the horizon for the stored energy `load_kw` needs.

    Returns an implementable reserve request and a separate feasibility diagnostic; initial_kwh does
    not imply that all forecast load can be served. The request saturates at capacity every hour (a
    pack cannot hold more), while an unsaturated twin recursion records the largest need anywhere.
    """
    eta = math.sqrt(efficiency)
    need = unlimited = peak = 0.0
    for pv, load in zip(reversed(pv_kw), reversed(load_kw), strict=True):
        net = pv - load
        need = max(0.0, need - net * eta) if net >= 0 else need - net / eta
        unlimited = max(0.0, unlimited - net * eta) if net >= 0 else unlimited - net / eta
        need = min(need, capacity_kwh)
        peak = max(peak, unlimited)
    return RequiredEnergy(initial_kwh=need, feasible=peak <= capacity_kwh + 1e-9, max_required_kwh=peak)


def probe_pump_kwh(
    headroom_kwh: float, eta: float, pv: float, critical: float, nc_served: float, pump_kw: float, tank_room: float
) -> float:
    """Pumping on a probe night: one AC balance, so PV and battery headroom net of the hour's loads."""
    available = headroom_kwh * eta + pv - critical - nc_served
    return max(0.0, min(pump_kw, tank_room, available))


def kalman_update(est: float, var: float, z: float, r: float) -> tuple[float, float]:
    """Standard scalar Kalman measurement update."""
    gain = var / (var + r)
    return est + gain * (z - est), (1.0 - gain) * var


@dataclass(frozen=True)
class Candidate:
    z: float
    r: float
    day: int
    sign: int


@dataclass(frozen=True)
class CapacityFilter:
    est: float
    var: float
    candidate: Candidate | None = None


def _clipped(f: CapacityFilter) -> tuple[CapacityFilter, bool]:
    lo, hi = CAPACITY_BOUNDS
    est = min(hi, max(lo, f.est))
    return CapacityFilter(est, f.var, f.candidate), est != f.est


def filter_predict(f: CapacityFilter, drift: float, process_var: float) -> tuple[CapacityFilter, bool]:
    """Daily prediction step: expected fade and process noise; returns the filter and whether it was clipped."""
    return _clipped(CapacityFilter(f.est - drift, f.var + process_var, f.candidate))


def filter_update(f: CapacityFilter, z: float, r: float, day: int) -> tuple[CapacityFilter, str, bool]:
    """Measurement update with two-sample step confirmation.

    Returns the filter, the event ("update", "held" or "jump") and whether the estimate was clipped.
    """
    nu = (z - f.est) / math.sqrt(f.var + r)
    cand = f.candidate if f.candidate and day - f.candidate.day <= CANDIDATE_LIFETIME_DAYS else None
    if abs(nu) <= JUMP_SIGMAS:
        est, var = kalman_update(f.est, f.var, z, r)
        new, clipped = _clipped(CapacityFilter(est, var))
        return new, "update", clipped
    sign = 1 if nu > 0 else -1
    if cand and cand.sign == sign and abs(z - cand.z) <= CONFIRM_SIGMAS * math.sqrt(cand.r + r):
        w1, w2 = 1.0 / cand.r, 1.0 / r
        est = (cand.z * w1 + z * w2) / (w1 + w2)
        new, clipped = _clipped(CapacityFilter(est, max(1.0 / (w1 + w2), JUMP_RESET_SIGMA**2)))
        return new, "jump", clipped
    return CapacityFilter(f.est, f.var, Candidate(z, r, day, sign)), "held", False


_TOTALS = (
    "demand", "served", "unmet_c", "unmet_nc", "pumped", "degree_hours", "precool", "spill_pump",
    "pv", "batt_in", "batt_out", "spilled", "fade_clip", "compressor", "leak", "from_tank",
)  # fmt: skip


class _State:
    """Mutable accumulator for the hot loop; never escapes `simulate`."""

    def __init__(self, pack: Pack, sensing: Sensing) -> None:
        self.cap = pack.initial_capacity
        self.energy = self.cap * pack.nominal_kwh
        self.est, self.var = sensing.prior_capacity, sensing.prior_sigma**2
        self.candidate: Candidate | None = None
        self.soc_est = 1.0
        self.last_anchor, self.anchor_soc, self.min_soc = "F", 1.0, 1.0
        self.empty_armed, self.was_full = True, True
        self.tank = 0.0
        self.irrigation_left = 0.0
        self.probe_night, self.last_probe_day = False, -(10**6)
        self.day, self.last_pair_day = 0, 0
        self.stepped = False
        self.step = 0
        self.forecast: list[float] = []
        self.forecast_start = 0
        self.step_day: int | None = None
        self.detect_days: float | None = None
        self.errors: list[float] = []
        self.cold = 0.0
        self.initial_energy = self.energy
        self.initial_cold = 0.0
        self.totals = dict.fromkeys(_TOTALS, 0.0)
        self.counts = dict.fromkeys(
            ("pairs", "partials", "empties", "probes", "hot_hours", "infeasible", "held", "jumps", "clips"), 0
        )


def simulate(
    site: Site,
    farm: Farm,
    pack: Pack,
    sensing: Sensing,
    policy: Policy,
    hours: int | None = None,
    start_hour: int = 0,
    seed: int = 0,
) -> Result:
    rng = random.Random(seed)
    s = _State(pack, sensing)
    s.cold = s.initial_cold = farm.cold_room.q_set if farm.cold_room else 0.0
    n = len(site.pv_kw_per_kwp)
    hours = n if hours is None else hours
    climatology = _climatology(site) if policy.forecast_climatology else None
    for step in range(hours):
        t = (start_hour + step) % n
        hour = step % HOURS_PER_DAY
        day = step // HOURS_PER_DAY
        s.step = step
        if hour == 0:
            _start_day(s, farm, sensing, day, start_hour + step)
        _apply_ageing(s, pack, site.ambient_c[t], step)
        if policy.forecast_reserve and (step == 0 or hour == NIGHT_START):
            _issue_forecast(s, site, policy, rng, t, climatology)
        if hour == NIGHT_START:
            _decide_probe(s, site, farm, pack, sensing, policy, rng, t, day)
        if hour == NIGHT_END:
            s.probe_night = False
        _dispatch(s, site, farm, pack, sensing, policy, t, hour)
        _observe(s, pack, sensing, rng)
    return _result(s)


def _climatology(site: Site) -> list[list[float]]:
    sums = [[0.0] * HOURS_PER_DAY for _ in range(12)]
    counts = [[0] * HOURS_PER_DAY for _ in range(12)]
    for i, pv in enumerate(site.pv_kw_per_kwp):
        m, h = _month(i), i % HOURS_PER_DAY
        sums[m][h] += pv
        counts[m][h] += 1
    return [[sums[m][h] / max(1, counts[m][h]) for h in range(HOURS_PER_DAY)] for m in range(12)]


def _issue_forecast(
    s: _State, site: Site, policy: Policy, rng: random.Random, t: int, climatology: list[list[float]] | None
) -> None:
    """Hourly PV per kWp for the next 72 h; per-lead-day lognormal error, then a pessimism haircut."""
    n = len(site.pv_kw_per_kwp)
    sigmas = policy.forecast_sigmas
    factors = [math.exp(rng.gauss(0.0, sg) - sg * sg / 2) for sg in sigmas]
    fc = []
    for k in range(FORECAST_HORIZON_H):
        lead = min(len(sigmas) - 1, k // HOURS_PER_DAY)
        haircut = math.exp(-policy.forecast_pessimism_z * sigmas[lead])
        i = (t + k) % n
        if climatology is not None:
            fc.append(climatology[_month(i)][i % HOURS_PER_DAY] * haircut)
        else:
            fc.append(site.pv_kw_per_kwp[i] * factors[lead] * haircut)
    s.forecast, s.forecast_start = fc, s.step


def _month(abs_hour: int) -> int:
    return (abs_hour // HOURS_PER_DAY % 365) * 12 // 365


def _start_day(s: _State, farm: Farm, sensing: Sensing, day: int, abs_hour: int) -> None:
    s.day = day
    f, clipped = filter_predict(
        CapacityFilter(s.est, s.var, s.candidate), sensing.drift_per_day, sensing.process_sigma_per_day**2
    )
    s.est, s.var, s.candidate = f.est, f.var, f.candidate
    s.counts["clips"] += clipped
    err = s.est - s.cap
    s.errors.append(err)
    if s.step_day is not None and s.detect_days is None and abs(err) < DETECT_TOLERANCE:
        s.detect_days = day - s.step_day
    need = farm.irrigation_kwh[_month(abs_hour)]
    from_tank = min(need, s.tank)
    s.tank -= from_tank
    s.totals["from_tank"] += from_tank
    s.irrigation_left = need - from_tank


def _apply_ageing(s: _State, pack: Pack, ambient_c: float, step: int) -> None:
    temp = ambient_c + pack.temp_offset_c
    s.cap -= pack.calendar_fade_per_year / HOURS_PER_YEAR * 2 ** ((temp - 25.0) / 10.0)
    if pack.step_hour is not None and not s.stepped and step >= pack.step_hour:
        s.cap -= pack.step_loss
        s.stepped, s.step_day = True, step // HOURS_PER_DAY
    clipped = max(0.0, s.energy - s.cap * pack.nominal_kwh)
    s.energy -= clipped
    s.totals["fade_clip"] += clipped


def _controller_capacity(s: _State, policy: Policy) -> float:
    return s.cap if policy.oracle else s.est


def _decide_probe(
    s: _State,
    site: Site,
    farm: Farm,
    pack: Pack,
    sensing: Sensing,
    policy: Policy,
    rng: random.Random,
    t: int,
    day: int,
) -> None:
    if not policy.probe or policy.oracle:
        return
    uncertain = math.sqrt(s.var) > policy.probe_trigger_sigma
    stale = policy.probe_max_gap_days is not None and day - s.last_pair_day > policy.probe_max_gap_days
    if not (uncertain or stale):
        return
    if day - s.last_probe_day < policy.probe_min_gap_days or s.tank >= farm.tank_kwh:
        return
    n = len(site.pv_kw_per_kwp)
    pv_next = farm.pv_kwp * sum(site.pv_kw_per_kwp[(t + k) % n] for k in range(1, HOURS_PER_DAY + 1))
    forecast = pv_next * math.exp(rng.gauss(0.0, policy.forecast_sigma))
    load_next = farm.critical_kw * HOURS_PER_DAY + sum(farm.household_kw) + max(farm.irrigation_kwh)
    refill = (1.0 - sensing.empty_soc) * _controller_capacity(s, policy) * pack.nominal_kwh
    if forecast >= policy.probe_forecast_margin * (load_next + refill):
        s.probe_night, s.last_probe_day = True, day
        s.counts["probes"] += 1


def _dispatch(
    s: _State, site: Site, farm: Farm, pack: Pack, sensing: Sensing, policy: Policy, t: int, hour: int
) -> None:
    pv = farm.pv_kwp * site.pv_kw_per_kwp[t]
    room, ambient = farm.cold_room, site.ambient_c[t]
    critical = compressor_need(room, s.cold, ambient, room.q_set) if room else farm.critical_kw
    irrigation = 0.0
    if hour in farm.irrigation_hours and s.irrigation_left > 0:
        irrigation = min(s.irrigation_left, s.irrigation_left / max(1, farm.irrigation_hours[-1] - hour + 1))
        s.irrigation_left -= irrigation
    noncritical = farm.household_kw[hour] + irrigation
    s.totals["demand"] += critical + noncritical

    threshold = _shedding_threshold(s, site, farm, pack, sensing, policy, t, hour)
    headroom = max(0.0, s.soc_est - threshold) * _controller_capacity(s, policy) * pack.nominal_kwh
    from_battery = headroom * math.sqrt(pack.round_trip_efficiency)
    nc_served = min(noncritical, max(0.0, pv - critical) + from_battery)
    pump = 0.0
    if s.probe_night and s.soc_est > threshold and s.tank < farm.tank_kwh:
        headroom = (s.soc_est - threshold) * _controller_capacity(s, policy) * pack.nominal_kwh
        eta = math.sqrt(pack.round_trip_efficiency)
        pump = probe_pump_kwh(headroom, eta, pv, critical, nc_served, farm.pump_kw, farm.tank_kwh - s.tank)

    eta = math.sqrt(pack.round_trip_efficiency)
    surplus = pv - critical - nc_served - pump
    soc_before = s.energy / (s.cap * pack.nominal_kwh)
    curtailed = 0.0
    s.totals["pv"] += pv
    if surplus >= 0:
        stored = min(surplus * eta, s.cap * pack.nominal_kwh - s.energy)
        s.energy += stored
        s.totals["batt_in"] += stored / eta
        curtailed = surplus - stored / eta
        shortfall = 0.0
    else:
        available = max(0.0, s.energy - pack.floor_soc * s.cap * pack.nominal_kwh) * eta
        delivered = min(-surplus, available)
        s.energy -= delivered / eta
        s.totals["batt_out"] += delivered
        shortfall = -surplus - delivered
    soc_after = s.energy / (s.cap * pack.nominal_kwh)
    if soc_after < soc_before:
        s.cap -= cycle_wear(soc_before, soc_after, pack.cycle_fade_per_full_cycle, pack.depth_exponent)

    pump_cut = min(pump, shortfall)
    shortfall -= pump_cut
    nc_cut = min(nc_served, shortfall)
    shortfall -= nc_cut
    s.tank += pump - pump_cut
    s.totals["pumped"] += pump - pump_cut
    s.totals["served"] += critical - shortfall + nc_served - nc_cut
    s.totals["unmet_c"] += shortfall
    s.totals["unmet_nc"] += noncritical - nc_served + nc_cut
    if room:
        _use_surplus(s, farm, room, policy, ambient, critical - shortfall, curtailed, pump + irrigation)
    else:
        s.totals["spilled"] += curtailed

    measured = (soc_after - soc_before) * s.cap * (1.0 + sensing.gain_error)
    s.soc_est = min(1.0, max(0.0, s.soc_est + measured / _controller_capacity(s, policy)))


def _use_surplus(
    s: _State,
    farm: Farm,
    room: ColdRoom,
    policy: Policy,
    ambient: float,
    cooling: float,
    curtailed: float,
    pumping: float,
) -> None:
    """Spend PV the battery could not take on pre-cooling, then on the water tank; update the room."""
    precool = 0.0
    if policy.precool_surplus and curtailed > 0:
        to_max = (room.q_max - s.cold + leak_kwh(room, ambient, s.cold)) / room.cop - cooling
        precool = max(0.0, min(curtailed, room.compressor_kw - cooling, to_max))
    spill_pump = 0.0
    if policy.pump_surplus and curtailed - precool > 0:
        spill_pump = max(0.0, min(curtailed - precool, farm.pump_kw - pumping, farm.tank_kwh - s.tank))
    s.tank += spill_pump
    s.totals["pumped"] += spill_pump
    s.totals["spill_pump"] += spill_pump
    s.totals["precool"] += precool
    s.totals["spilled"] += curtailed - precool - spill_pump
    s.totals["compressor"] += cooling + precool
    leak = leak_kwh(room, ambient, s.cold)
    s.totals["leak"] += leak
    s.cold += (cooling + precool) * room.cop - leak
    temp = temperature(room, s.cold)
    if temp > room.t_max:
        s.counts["hot_hours"] += 1
        s.totals["degree_hours"] += temp - room.t_max


def _shedding_threshold(
    s: _State, site: Site, farm: Farm, pack: Pack, sensing: Sensing, policy: Policy, t: int, hour: int
) -> float:
    if not s.probe_night:
        if policy.forecast_reserve:
            return _forecast_reserve(s, site, farm, pack, policy, t)
        return policy.reserve_soc
    if not policy.probe_dawn_timed:
        return sensing.empty_soc + PROBE_SOC_MARGIN
    hours_left = (PV_COVERS_LOAD_HOUR - hour) % HOURS_PER_DAY
    cold_room = farm.critical_kw * hours_left / (_controller_capacity(s, policy) * pack.nominal_kwh)
    return sensing.empty_soc - DAWN_LANDING_BELOW_EMPTY + cold_room


def _forecast_reserve(s: _State, site: Site, farm: Farm, pack: Pack, policy: Policy, t: int) -> float:
    capacity = _controller_capacity(s, policy) * pack.nominal_kwh
    pv = [farm.pv_kwp * x for x in s.forecast[s.step - s.forecast_start :]]
    room = farm.cold_room
    if room:
        n = len(site.ambient_c)
        ambient = [site.ambient_c[(t + k) % n] for k in range(len(pv))]
        credit = max(0.0, s.cold - room.q_set) if policy.cold_credit else 0.0
        loads = planned_loads(room, ambient, credit)
    else:
        loads = [farm.critical_kw] * len(pv)
    req = required_energy(pv, loads, capacity, pack.round_trip_efficiency)
    if not req.feasible:
        s.counts["infeasible"] += 1
    return min(MAX_RESERVE_SOC, pack.floor_soc + req.initial_kwh / capacity + policy.reserve_margin)


def _observe(s: _State, pack: Pack, sensing: Sensing, rng: random.Random) -> None:
    soc = s.energy / (s.cap * pack.nominal_kwh)
    s.min_soc = min(s.min_soc, soc)
    if soc > REARM_SOC:
        s.empty_armed = True
    if soc <= sensing.empty_soc and s.empty_armed:
        s.empty_armed = False
        s.counts["empties"] += 1
        if s.last_anchor == "F":
            _sample_pair(s, sensing, rng, span=1.0 - soc)
        s.last_anchor, s.anchor_soc = "E", soc
        s.soc_est = min(1.0, max(0.0, soc + rng.gauss(0.0, 0.01)))
    is_full = soc >= FULL_SOC
    if is_full and not s.was_full:
        if s.last_anchor == "E":
            _sample_pair(s, sensing, rng, span=1.0 - s.anchor_soc)
        elif s.min_soc <= PARTIAL_START_SOC and sensing.partial_sigma is not None:
            z = s.cap * (1.0 + sensing.gain_error) * (1.0 + rng.gauss(0.0, sensing.partial_sigma))
            _measure(s, z, sensing.partial_sigma**2)
            s.counts["partials"] += 1
        s.last_anchor, s.min_soc, s.soc_est = "F", 1.0, 1.0
    s.was_full = is_full


def _sample_pair(s: _State, sensing: Sensing, rng: random.Random, span: float) -> None:
    sigma = sensing.pair_sigma * 0.9 / max(span, 0.3)
    z = s.cap * (1.0 + sensing.gain_error) * (1.0 + rng.gauss(0.0, sigma))
    _measure(s, z, sigma**2)
    s.counts["pairs"] += 1
    s.last_pair_day = s.day


def _measure(s: _State, z: float, r: float) -> None:
    f, event, clipped = filter_update(CapacityFilter(s.est, s.var, s.candidate), z, r, s.day)
    s.est, s.var, s.candidate = f.est, f.var, f.candidate
    s.counts["held"] += event == "held"
    s.counts["jumps"] += event == "jump"
    s.counts["clips"] += clipped


def _result(s: _State) -> Result:
    errs = s.errors or [s.est - s.cap]
    abs_sorted = sorted(abs(e) for e in errs)
    return Result(
        demand_kwh=s.totals["demand"],
        served_kwh=s.totals["served"],
        unmet_critical_kwh=s.totals["unmet_c"],
        unmet_noncritical_kwh=s.totals["unmet_nc"],
        pumped_kwh=s.totals["pumped"],
        final_capacity=s.cap,
        final_abs_error=abs(s.est - s.cap),
        rmse_error=math.sqrt(sum(e * e for e in errs) / len(errs)),
        p95_abs_error=abs_sorted[min(len(abs_sorted) - 1, int(0.95 * len(abs_sorted)))],
        detect_days=s.detect_days,
        pair_samples=s.counts["pairs"],
        partial_samples=s.counts["partials"],
        empty_anchors=s.counts["empties"],
        probes=s.counts["probes"],
        hours_above_max=s.counts["hot_hours"],
        degree_hours_above_max=s.totals["degree_hours"],
        precool_kwh=s.totals["precool"],
        spill_pumped_kwh=s.totals["spill_pump"],
        pv_kwh=s.totals["pv"],
        battery_in_kwh=s.totals["batt_in"],
        battery_out_kwh=s.totals["batt_out"],
        spilled_kwh=s.totals["spilled"],
        fade_clipped_kwh=s.totals["fade_clip"],
        initial_energy_kwh=s.initial_energy,
        final_energy_kwh=s.energy,
        compressor_kwh=s.totals["compressor"],
        cold_leak_kwh_th=s.totals["leak"],
        initial_cold_kwh_th=s.initial_cold,
        final_cold_kwh_th=s.cold,
        tank_drawn_kwh=s.totals["from_tank"],
        final_tank_kwh=s.tank,
        infeasible_horizons=s.counts["infeasible"],
        held_outliers=s.counts["held"],
        confirmed_jumps=s.counts["jumps"],
        estimate_clips=s.counts["clips"],
    )
