"""Cold room as a thermal store: temperature, leak and compressor need, checkable by hand."""

import pytest

from farmsim.coldroom import ColdRoom, compressor_need, leak_kwh, planned_loads, temperature

ROOM = ColdRoom(ua_kw_per_k=0.05, capacity_kwh_per_k=0.5, cop=2.0, compressor_kw=1.0)


def test_setpoint_store_reads_setpoint_and_empty_store_reads_the_limit():
    assert temperature(ROOM, ROOM.q_set) == pytest.approx(ROOM.t_set)
    assert temperature(ROOM, 0.0) == pytest.approx(ROOM.t_max)
    assert temperature(ROOM, ROOM.q_max) == pytest.approx(ROOM.t_min)


def test_bank_holds_setpoint_while_it_melts_then_room_warms():
    banked = ColdRoom(bank_kwh=3.0)
    assert temperature(banked, banked.q_set - 1.5) == pytest.approx(banked.t_set)
    assert temperature(banked, 0.0) == pytest.approx(banked.t_max)
    assert temperature(banked, -0.5) > banked.t_max


def test_leak_grows_as_the_room_gets_colder():
    assert leak_kwh(ROOM, ambient_c=24.0, q=ROOM.q_max) > leak_kwh(ROOM, ambient_c=24.0, q=ROOM.q_set)
    assert leak_kwh(ROOM, ambient_c=24.0, q=ROOM.q_set) == pytest.approx(0.05 * 20)


def test_thermostat_need_replaces_the_leak_and_is_capped_by_the_compressor():
    assert compressor_need(ROOM, ROOM.q_set, 24.0, ROOM.q_set) == pytest.approx(1.0 / 2.0)
    assert compressor_need(ROOM, -10.0, 24.0, ROOM.q_set) == pytest.approx(1.0)
    assert compressor_need(ROOM, ROOM.q_max, 24.0, ROOM.q_set) == 0.0


def test_planned_loads_spend_stored_cold_first():
    plain = planned_loads(ROOM, [24.0] * 4, credit_kwh_th=0.0)
    credited = planned_loads(ROOM, [24.0] * 4, credit_kwh_th=1.5)
    assert plain == pytest.approx([0.5] * 4)
    assert credited == pytest.approx([0.0, 0.25, 0.5, 0.5])
