"""Tolerance snapshots: any change to these outputs must be intended and reviewed (see snapshot_cases.py)."""

import json
import math
import os
from pathlib import Path

import pytest
from snapshot_cases import CASES, run_all, run_case

SNAPSHOTS = Path(__file__).with_name("snapshots.json")
RTOL, ATOL = 1e-6, 1e-9


def _close(a: object, b: object) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


@pytest.fixture(scope="module")
def stored() -> dict:
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOTS.write_text(json.dumps(run_all(), indent=1, sort_keys=True) + "\n")
    return json.loads(SNAPSHOTS.read_text())


@pytest.mark.parametrize("name", sorted(CASES))
def test_outputs_match_snapshot(stored: dict, name: str):
    current, expected = run_case(name), stored[name]
    changed = {k: (expected.get(k), v) for k, v in current.items() if not _close(expected.get(k), v)}
    assert not changed, f"{name}: (snapshot, now) {changed}"
