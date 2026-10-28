"""Cold room as a thermal store (experiment 3).

State `q` is the cold stored below the spoilage limit `t_max`, in kWh of heat: q = 0 at `t_max`,
negative when the room is warmer. Depletion order: pre-cooling below the setpoint first, then the
bank (ice or PCM, which holds the setpoint while it melts), then the room warms towards and past
`t_max`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColdRoom:
    ua_kw_per_k: float = 0.041  # heat leak; ~0.45 kW electric on average at 26 C ambient
    capacity_kwh_per_k: float = 0.5  # produce and structure
    cop: float = 2.0
    compressor_kw: float = 0.9  # electric
    t_set: float = 4.0
    t_min: float = 1.0
    t_max: float = 8.0
    bank_kwh: float = 0.0  # thermal

    @property
    def q_set(self) -> float:
        return self.capacity_kwh_per_k * (self.t_max - self.t_set) + self.bank_kwh

    @property
    def q_max(self) -> float:
        return self.capacity_kwh_per_k * (self.t_max - self.t_min) + self.bank_kwh


def temperature(room: ColdRoom, q: float) -> float:
    if q >= room.q_set:
        return room.t_set - (q - room.q_set) / room.capacity_kwh_per_k
    if q >= room.q_set - room.bank_kwh:
        return room.t_set
    return room.t_set + (room.q_set - room.bank_kwh - q) / room.capacity_kwh_per_k


def leak_kwh(room: ColdRoom, ambient_c: float, q: float) -> float:
    """Heat entering in one hour."""
    return max(0.0, room.ua_kw_per_k * (ambient_c - temperature(room, q)))


def compressor_need(room: ColdRoom, q: float, ambient_c: float, target_q: float) -> float:
    """Electric energy this hour to reach `target_q` after the hour's leak, capped by the compressor."""
    need = (target_q - q + leak_kwh(room, ambient_c, q)) / room.cop
    return min(room.compressor_kw, max(0.0, need))


def planned_loads(room: ColdRoom, ambient_c: list[float], credit_kwh_th: float) -> list[float]:
    """Electric load per hour to hold the setpoint, with stored cold above it spent first."""
    credit = credit_kwh_th / room.cop
    loads = []
    for amb in ambient_c:
        load = max(0.0, room.ua_kw_per_k * (amb - room.t_set)) / room.cop
        use = min(credit, load)
        credit -= use
        loads.append(load - use)
    return loads
