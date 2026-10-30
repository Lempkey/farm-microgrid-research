"""The experiment-1 headline check must refuse incomplete or unpaired rows rather than pass on missing evidence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from gate_c100 import IncompleteRows, headline  # noqa: E402


def rows(seeds=range(3), policies=("reserve", "oracle"), cases=(("base", "used", "4pp"),), unmet=None):
    unmet = unmet or {"reserve": 10.0, "oracle": 9.9}
    return [
        dict(farm=f, pack=k, partial=p, policy=pol, seed=s, unmet_critical_kwh=unmet[pol])
        for f, k, p in cases
        for pol in policies
        for s in seeds
    ]


def test_complete_paired_rows_give_the_largest_relative_change():
    table, worst = headline(rows())
    assert len(table) == 1
    assert str(worst) == "1.0"


def test_missing_passive_rows_fail_instead_of_skipping_the_case():
    with pytest.raises(IncompleteRows, match="reserve"):
        headline(rows(policies=("oracle",)))


def test_unpaired_seed_sets_fail():
    data = [r for r in rows() if not (r["policy"] == "oracle" and r["seed"] == 2)]
    with pytest.raises(IncompleteRows, match="seed"):
        headline(data)


def test_duplicate_rows_fail():
    data = rows()
    with pytest.raises(IncompleteRows, match="duplicate"):
        headline(data + data[:1])


def test_expected_case_and_seed_counts_are_enforced():
    with pytest.raises(IncompleteRows, match="cases"):
        headline(rows(), expected_cases=12)
    with pytest.raises(IncompleteRows, match="seeds"):
        headline(rows(), expected_seeds=20)
    _, worst = headline(rows(), expected_cases=1, expected_seeds=3)
    assert str(worst) == "1.0"


def test_case_with_no_passive_unmet_load_is_reported_not_counted():
    table, worst = headline(rows(unmet={"reserve": 0.0, "oracle": 0.0}))
    assert table[0][2] is None
    assert str(worst) == "0.0"
