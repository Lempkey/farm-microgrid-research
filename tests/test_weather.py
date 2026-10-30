"""Weather years are refused unless complete: hourly, ordered, gap-free, whole calendar years, no fill values."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from common import IncompleteWeather, validated_year  # noqa: E402


def year_keys(year: int) -> list[str]:
    t, end, keys = datetime(year, 1, 1), datetime(year + 1, 1, 1), []
    while t < end:
        keys.append(t.strftime("%Y%m%d%H"))
        t += timedelta(hours=1)
    return keys


def parameter(year: int, ghi: float = 100.0, temp: float = 26.0) -> dict:
    keys = year_keys(year)
    return {"ALLSKY_SFC_SW_DWN": dict.fromkeys(keys, ghi), "T2M": dict.fromkeys(keys, temp)}


@pytest.mark.parametrize("year, hours", [(2023, 8760), (2024, 8784)])
def test_complete_year_is_accepted_with_its_calendar_length(year, hours):
    keys, ghi, temp = validated_year(year, parameter(year))
    assert len(keys) == len(ghi) == len(temp) == hours
    assert keys[0] == f"{year}010100" and keys[-1] == f"{year}123123"


def test_truncated_year_is_refused():
    p = parameter(2023)
    for k in year_keys(2023)[-48:]:
        del p["ALLSKY_SFC_SW_DWN"][k], p["T2M"][k]
    with pytest.raises(IncompleteWeather, match="8712 hours"):
        validated_year(2023, p)


def test_gap_inside_the_year_is_refused_even_with_the_right_count():
    p, keys = parameter(2023), year_keys(2023)
    for name in p:
        del p[name][keys[100]]
        p[name]["2024010100"] = 1.0
    with pytest.raises(IncompleteWeather, match="not the hours of 2023"):
        validated_year(2023, p)


def test_mismatched_parameter_timestamps_are_refused():
    p = parameter(2023)
    del p["T2M"][year_keys(2023)[5]]
    with pytest.raises(IncompleteWeather, match="timestamps differ"):
        validated_year(2023, p)


@pytest.mark.parametrize("name", ["ALLSKY_SFC_SW_DWN", "T2M"])
def test_fill_values_are_refused(name):
    p = parameter(2023)
    p[name][year_keys(2023)[10]] = -999.0
    with pytest.raises(IncompleteWeather, match="fill value"):
        validated_year(2023, p)
