"""Summarise the per-year re-run (per_year.py): every headline of experiments 2, 2b and 3 per weather year.

For each year, seeds are averaged per policy, the reference family's Pareto envelope is built, and the
challenger (a point or a frontier) is compared on the common support (farmsim.frontier). Years outside
the support are counted as dominates / dominated / no overlap and left out of the means. Across years:
the mean absolute gap and the ratio of sums, with circular block-bootstrap intervals (block 2 primary,
3 and iid as sensitivities; 5,000 resamples). Wear over 5 years comes from the four 5-year windows,
shown individually. Writes a markdown report.

Usage: python experiments/summarise_per_year.py <b1_rows.json> <report.md>
"""

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from farmsim.frontier import Comparison, compare_frontiers, compare_point, pareto_envelope
from farmsim.stats import block_bootstrap, percentile_interval

RESAMPLES, BOOT_SEED = 5000, 2026
BLOCKS = {"block 2": 2, "block 3": 3, "iid": 1}
SMALL_REFERENCE = 1.0  # kWh/yr or h/yr: below this a percentage is n/a
CONFIRM, WEAK = 0.25, 0.10  # the R-44/R-45 rule thresholds
EXP2_FIXED = [f"fixed_{x:02d}" for x in (5, 20, 35, 50, 65, 80)]
MARGIN_TAGS = ("00", "02", "05", "10", "15", "25")
EXP3_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25)


@dataclass(frozen=True)
class Summary:
    years: list[tuple[int, Comparison]]
    mean_gap: float | None
    gap_ci: tuple[float, float] | None
    ratio: float | None
    ratio_ci: dict[str, tuple[float, float]]
    kinds: Counter


def seed_means(rows: list[dict], metric: str) -> dict[tuple, tuple[float, float]]:
    """(pack, year, policy) -> (mean unmet non-critical, mean metric) per year of run."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["pack"], r["year"], r["policy"])].append(
            (r["unmet_noncritical_kwh"] / r["n_years"], r[metric] / r["n_years"])
        )
    return {k: (st.mean(x for x, _ in v), st.mean(y for _, y in v)) for k, v in acc.items()}


def summarise(comparisons: list[tuple[int, Comparison]]) -> Summary:
    gaps = [(c.gap, c.reference) for _, c in comparisons if c.kind == "gap"]
    kinds = Counter(c.kind for _, c in comparisons)
    if not gaps:
        return Summary(comparisons, None, None, None, {}, kinds)
    mean_gap = st.mean(g for g, _ in gaps)
    gap_ci = percentile_interval(block_bootstrap(gaps, lambda xs: st.mean(g for g, _ in xs), 2, RESAMPLES, BOOT_SEED))
    ratio, ratio_ci = None, {}
    if st.mean(r for _, r in gaps) >= SMALL_REFERENCE:
        ratio = sum(g for g, _ in gaps) / sum(r for _, r in gaps)
        for name, block in BLOCKS.items():
            stat = _ratio if block <= len(gaps) else None
            if stat:
                ratio_ci[name] = percentile_interval(block_bootstrap(gaps, stat, block, RESAMPLES, BOOT_SEED))
    return Summary(comparisons, mean_gap, gap_ci, ratio, ratio_ci, kinds)


def _ratio(xs: list[tuple[float, float]]) -> float:
    ref = sum(r for _, r in xs)
    return sum(g for g, _ in xs) / ref if ref > 0 else 0.0


def per_year(means: dict, pack: str, years: list[int], reference: list[str], challenger: list[str]) -> list:
    out = []
    for y in years:
        ref = pareto_envelope([means[(pack, y, p)] for p in reference])
        if len(challenger) == 1:
            out.append((y, compare_point(ref, means[(pack, y, challenger[0])])))
        else:
            out.append((y, compare_frontiers(ref, pareto_envelope([means[(pack, y, p)] for p in challenger]))))
    return out


def category(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    return "confirmed (>= 25%)" if ratio >= CONFIRM else "weak (10-25%)" if ratio >= WEAK else "dropped (< 10%)"


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.0f}%"


def report_block(title: str, unit: str, s: Summary, original: str) -> list[str]:
    lines = [f"\n### {title}\n", f"Original: {original}\n"]
    kinds = ", ".join(f"{k} {n}" for k, n in sorted(s.kinds.items()))
    lines.append(f"- Years: {kinds}")
    if s.mean_gap is None:
        lines.append("- No year with common support.")
        return lines
    lo, hi = s.gap_ci
    lines.append(f"- **Mean gap: {s.mean_gap:+.2f} {unit}** (block-2 95% interval {lo:+.2f} to {hi:+.2f})")
    if s.ratio is None:
        lines.append(f"- Ratio: n/a (mean reference below {SMALL_REFERENCE} {unit})")
    else:
        cis = "; ".join(f"{k} {fmt_pct(a)} to {fmt_pct(b)}" for k, (a, b) in s.ratio_ci.items())
        lines.append(f"- **Ratio of sums: {fmt_pct(s.ratio)}** ({cis}); rule category: {category(s.ratio)}")
    cells = []
    for y, c in s.years:
        if c.kind != "gap":
            cells.append(f"{y}: {c.kind}")
        else:
            pct = "" if c.reference < SMALL_REFERENCE else f" ({fmt_pct(c.gap / c.reference)})"
            cells.append(f"{y}: {c.gap:+.2f}{pct}")
    lines.append("- Per year (gap" + (", % of reference" if s.ratio is not None else "") + "): " + "; ".join(cells))
    return lines


def window_wear(rows: list[dict], exp: str, farm: str, pack: str, base: list[str], other: list[str]) -> dict:
    """Extra capacity lost by `other` against `base` over each 5-year window, in percentage points."""
    caps = defaultdict(list)
    for r in rows:
        mine = r["unit"] == "window" and r["exp"] == exp and r["farm"] == farm and r["pack"] == pack
        if mine and r["policy"] in base + other:
            caps[(r["year"], r["policy"] in other)].append(r["final_capacity"])
    starts = sorted({y for y, _ in caps})
    return {y: 100 * (st.mean(caps[(y, False)]) - st.mean(caps[(y, True)])) for y in starts}


def fmt_wear(w: dict[int, float]) -> str:
    return ", ".join(f"{y}-{(y + 4) % 100:02d}: {v:+.2f} pp" for y, v in w.items())


def main() -> None:
    rows = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    years = sorted({r["year"] for r in rows if r["unit"] == "year"})
    by_exp = defaultdict(list)
    for r in rows:
        if r["unit"] == "year":
            by_exp[(r["exp"], r["farm"])].append(r)
    L = [
        "# Per-year re-run of experiments 2, 2b and 3",
        f"\n{len(years)} weather years ({years[0]}-{years[-1]}), each a 1-year run; seeds averaged within the "
        f"year; gaps on the common support of Pareto envelopes; intervals by circular block bootstrap "
        f"({RESAMPLES} resamples). Positive gap = challenger better. Placeholder loads, one site.",
    ]
    verdicts = []
    budgets = []

    L.append("\n## Experiment 2: need-sized reserve against the fixed-reserve frontier (unmet critical load)")
    for farm in ("base", "tight"):
        means = seed_means(by_exp[("2", farm)], "unmet_critical_kwh")
        for pack in ("new", "used"):
            budgets.append(
                (f"exp 2 fixed, {farm}, {pack}", [means[(pack, y, p)][0] for y in years for p in EXP2_FIXED])
            )
            res = {}
            for pol, orig in (
                ("fc_z1", "realistic forecast z = 1 (primary): base +43% new / +67% used"),
                ("fc_climatology", "climatology: base +27% new / +56% used"),
                ("fc_perfect", "perfect weather: base +31% new / +59% used; the rule needs >= 40%"),
                ("fc_z0", "z = 0: base +34% new / +61% used"),
                ("fc_z2", "z = 2: base +66% new / +80% used"),
            ):
                res[pol] = summarise(per_year(means, pack, years, EXP2_FIXED, [pol]))
                L += report_block(f"{farm} sizing, {pack} pack: {pol}", "kWh/yr", res[pol], orig)
            if farm == "base":
                wear = window_wear(rows, "2", "base", pack, ["fixed_35"], ["fc_z1"])
                L.append(f"\nWear of fc_z1 against fixed_35 over 5 years ({pack}): {fmt_wear(wear)}")
                ok = (
                    res["fc_z1"].ratio is not None
                    and res["fc_z1"].ratio >= CONFIRM
                    and (res["fc_perfect"].ratio or 0) >= 0.40
                    and all(w <= 1.0 for w in wear.values())
                )
                verdicts.append(
                    f"R-44 rule, base, {pack} pack: z = 1 {category(res['fc_z1'].ratio)}, perfect "
                    f"{fmt_pct(res['fc_perfect'].ratio)} (needs >= 40%), max window wear "
                    f"{max(wear.values()):+.2f} pp -> {'CONFIRMED' if ok else 'not confirmed by the letter'}"
                )

    L.append("\n## Experiment 2b (post hoc): perfect weather against climatology, frontier against frontier")
    means = seed_means(by_exp[("2b", "base")], "unmet_critical_kwh")
    for pack in ("new", "used"):
        s = summarise(
            per_year(means, pack, years, [f"clim_m{m}" for m in MARGIN_TAGS], [f"perfect_m{m}" for m in MARGIN_TAGS])
        )
        L += report_block(f"base, {pack} pack: perfect vs climatology", "kWh/yr", s, "only about 5-15% (C-104)")

    L.append("\n## Experiment 3: multi-store against the battery-only frontier (hours above 8 C)")
    rows3 = by_exp[("3", "base")]
    for metric, unit in (("hours_above_max", "h/yr"), ("degree_hours_above_max", "K h/yr")):
        means = seed_means(rows3, metric)
        L.append(f"\n**Metric: {metric}**")
        for pack in ("new", "used"):
            for room in ("produce", "bank"):
                ref = [f"{room}/batt_only/m{m:.2f}" for m in EXP3_MARGINS]
                if metric == "hours_above_max" and room == "produce":
                    budgets.append(
                        (f"exp 3 batt_only, produce, {pack}", [means[(pack, y, p)][0] for y in years for p in ref])
                    )
                for fam, orig in (
                    ("multi", "new: dominates; used: +82% (C-106)"),
                    ("cold_only", "+14 to +31% new, +23 to +30% used (C-107)"),
                    ("water_only", "new: dominates; used: +76%"),
                ):
                    ch = [f"{room}/{fam}/m{m:.2f}" for m in EXP3_MARGINS]
                    s = summarise(per_year(means, pack, years, ref, ch))
                    L += report_block(f"{pack} pack, {room} room: {fam}", unit, s, orig)
                    if metric == "hours_above_max" and room == "produce" and fam == "multi":
                        wear = window_wear(rows, "3", "base", pack, ref, ch)
                        L.append(f"\nWear of multi against batt_only over 5 years ({pack}): {fmt_wear(wear)}")
                        verdicts.append(
                            f"R-45 rule, produce room, {pack} pack: multi {category(s.ratio)} "
                            f"(ratio {fmt_pct(s.ratio)}; dominates in {s.kinds.get('dominates', 0)} of "
                            f"{len(years)} years), max window wear {max(wear.values()):+.2f} pp"
                        )
            big = [f"bigbatt/batt_only/m{m:.2f}" for m in EXP3_MARGINS]
            bank = [f"bank/batt_only/m{m:.2f}" for m in EXP3_MARGINS]
            s = summarise(per_year(means, pack, years, big, bank))
            L += report_block(
                f"{pack} pack, secondary: 6 kWh_th PCM bank vs a battery 3 kWh bigger", unit, s, "51-78% (C-108)"
            )

    L.append("\n## Verdicts by the original rules\n")
    L += [f"- {v}" for v in verdicts]
    L.append("\n## Reference families' realised unmet non-critical load (kWh/yr, seed means per year)\n")
    L += [f"- {name}: {min(xs):.1f} to {max(xs):.1f}" for name, xs in budgets]
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[-(len(verdicts) + len(budgets) + 4) :]))


if __name__ == "__main__":
    main()
