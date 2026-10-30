"""Experiment 1 headline: how much does exact capacity knowledge change unmet critical load?

For each case (farm, pack, partial-charge noise), compares passive learning ("reserve") with the
oracle, summed over seeds. Reports the largest relative change to one decimal place, rounding half up.
Cases where passive learning leaves no unmet critical load have no relative change and are reported as
n/a. Incomplete or unpaired rows are refused: every case needs both policies, the same seeds, and one row
per (case, policy, seed).

Usage: python experiments/gate_c100.py <exp1_rows.json> [--cases N] [--seeds N]
The full experiment 1 has 12 cases and 20 seeds.
"""

import argparse
import json
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal

YEARS = 5
POLICIES = ("reserve", "oracle")


class IncompleteRows(ValueError):
    pass


def _validate(rows: list[dict], expected_cases: int | None, expected_seeds: int | None) -> dict:
    """Seed set per case, after checking the rows are complete, paired and unique."""
    counts = Counter((r["farm"], r["pack"], r["partial"], r["policy"], r["seed"]) for r in rows)
    duplicates = [k for k, n in counts.items() if n > 1 and k[3] in POLICIES]
    if duplicates:
        raise IncompleteRows(f"duplicate rows: {duplicates[:3]}")
    seeds = defaultdict(lambda: {p: set() for p in POLICIES})
    for case_policy_seed in counts:
        *case, policy, seed = case_policy_seed
        if policy in POLICIES:
            seeds[tuple(case)][policy].add(seed)
    for case, by_policy in seeds.items():
        missing = [p for p in POLICIES if not by_policy[p]]
        if missing:
            raise IncompleteRows(f"{case}: no {', '.join(missing)} rows")
        if by_policy["reserve"] != by_policy["oracle"]:
            raise IncompleteRows(f"{case}: reserve and oracle seed sets differ")
    if expected_cases is not None and len(seeds) != expected_cases:
        raise IncompleteRows(f"{len(seeds)} cases, expected {expected_cases}")
    for case, by_policy in seeds.items():
        if expected_seeds is not None and len(by_policy["reserve"]) != expected_seeds:
            raise IncompleteRows(f"{case}: {len(by_policy['reserve'])} seeds, expected {expected_seeds}")
    return {case: by_policy["reserve"] for case, by_policy in seeds.items()}


def headline(
    rows: list[dict], expected_cases: int | None = None, expected_seeds: int | None = None
) -> tuple[list[tuple], Decimal]:
    seeds = _validate(rows, expected_cases, expected_seeds)
    unmet = defaultdict(lambda: dict.fromkeys(POLICIES, 0.0))
    for r in rows:
        if r["policy"] in POLICIES:
            unmet[(r["farm"], r["pack"], r["partial"])][r["policy"]] += r["unmet_critical_kwh"]
    table, worst = [], Decimal("0.0")
    for key in sorted(unmet):
        passive, oracle = unmet[key]["reserve"], unmet[key]["oracle"]
        per_year = (passive - oracle) / YEARS / len(seeds[key])
        if passive <= 0:
            table.append((key, per_year, None))
            continue
        rel = Decimal(str(100 * abs(passive - oracle) / passive)).quantize(Decimal("0.1"), ROUND_HALF_UP)
        worst = max(worst, rel)
        table.append((key, per_year, rel))
    return table, worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", nargs="+")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    for path in args.rows:
        table, worst = headline(json.load(open(path)), args.cases, args.seeds)
        print(f"\n{path}: {len(table)} cases x {args.seeds} seeds, complete and paired")
        print("| farm | pack | partial | oracle gain (kWh/yr) | change |\n|---|---|---|---|---|")
        for (farm, pack, partial), per_year, rel in table:
            print(f"| {farm} | {pack} | {partial} | {per_year:+.2f} | {'n/a' if rel is None else f'{rel}%'} |")
        print(f"largest change: {worst}%")


if __name__ == "__main__":
    main()
