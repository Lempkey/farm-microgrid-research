"""Markdown delta table between two regression snapshot files (see tests/snapshot_cases.py).

Usage: python experiments/snapshot_delta.py <old.json> <new.json>
"""

import json
import math
import sys

RTOL = 1e-6


def changed(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return not math.isclose(a, b, rel_tol=RTOL, abs_tol=1e-9)
    return a != b


def fmt(x: object) -> str:
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def main() -> None:
    old, new = (json.load(open(p)) for p in sys.argv[1:3])
    print("| case | output | before | after |\n|---|---|---|---|")
    for case in sorted(new):
        for key in sorted(new[case]):
            a, b = old.get(case, {}).get(key), new[case][key]
            if changed(a, b):
                print(f"| {case} | {key} | {fmt(a)} | {fmt(b)} |")


if __name__ == "__main__":
    main()
