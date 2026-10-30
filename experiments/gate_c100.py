"""Experiment 1 headline: how much does exact capacity knowledge change unmet critical load?

For each case (farm, pack, partial-charge noise), compares passive learning ("reserve") with the
oracle, summed over seeds. Reports the largest relative change to one decimal place, rounding half up.
Cases where passive learning leaves no unmet critical load have no relative change and are skipped.

Usage: python experiments/gate_c100.py <exp1_rows.json> [more_rows.json ...]
"""

import json
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

YEARS = 5


def headline(rows: list[dict]) -> tuple[list[tuple], Decimal]:
    unmet = defaultdict(lambda: defaultdict(float))
    seeds = defaultdict(set)
    for r in rows:
        key = (r["farm"], r["pack"], r["partial"])
        unmet[key][r["policy"]] += r["unmet_critical_kwh"]
        seeds[key].add(r["seed"])
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
    for path in sys.argv[1:]:
        table, worst = headline(json.load(open(path)))
        print(f"\n{path}\n| farm | pack | partial | oracle gain (kWh/yr) | change |\n|---|---|---|---|---|")
        for (farm, pack, partial), per_year, rel in table:
            print(f"| {farm} | {pack} | {partial} | {per_year:+.2f} | {'n/a' if rel is None else f'{rel}%'} |")
        print(f"largest change: {worst}%")


if __name__ == "__main__":
    main()
