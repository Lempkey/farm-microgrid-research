"""Summarise the per-year re-run (per_year.py): every headline of experiments 2, 2b and 3 per weather year.

For each year, seeds are averaged per policy, the reference family's Pareto envelope is built, and the
challenger (a point or a frontier) is compared on the common support (farmsim.frontier). Every year gets a
category: improvement or loss on common support, dominates, dominated, or no overlap. The mean gap and the
ratio of sums are *conditional on common support* and say so. Their intervals resample all years, in
order, in circular blocks (block 2 primary; block 3 and iid as sensitivities; 5,000 resamples); the
statistic then uses the resampled years that have common support, so blocks always join adjacent years.
Years are consecutive observations of one climate record, not independent draws. Wear and the module
loss come from the four 5-year windows, shown individually without intervals. Writes a markdown report.

Usage: python experiments/summarise_per_year.py <b1_rows.json> <report.md>
"""

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from farmsim.frontier import Comparison, compare_frontiers, compare_point, pareto_envelope
from farmsim.stats import block_bootstrap, percentile_interval

RESAMPLES, BOOT_SEED = 5000, 2026
MIN_YEARS_FOR_INTERVAL = 3  # fewer years with common support get no interval
PACK = {"new": "new pack", "used": "initially degraded pack"}  # annual runs: no module loss
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
    """Conditional mean gap and ratio over years with common support; intervals resample all years in order."""
    records = [(c.gap, c.reference) if c.kind == "gap" else None for _, c in comparisons]
    gaps = [r for r in records if r is not None]
    kinds = Counter(c.kind for _, c in comparisons)
    if not gaps:
        return Summary(comparisons, None, None, None, {}, kinds)
    mean_gap = _mean_gap(records)
    ratio = _ratio(records) if st.mean(r for _, r in gaps) >= SMALL_REFERENCE else None
    if len(gaps) < MIN_YEARS_FOR_INTERVAL:
        return Summary(comparisons, mean_gap, None, ratio, {}, kinds)
    gap_ci = _interval(records, _mean_gap, BLOCKS["block 2"])
    ratio_ci = {name: _interval(records, _ratio, b) for name, b in BLOCKS.items()} if ratio is not None else {}
    return Summary(comparisons, mean_gap, gap_ci, ratio, ratio_ci, kinds)


def _supported(xs: list) -> list[tuple[float, float]]:
    return [x for x in xs if x is not None]


def _mean_gap(xs: list) -> float:
    s = _supported(xs)
    return st.mean(g for g, _ in s) if s else math.nan


def _ratio(xs: list) -> float:
    s = _supported(xs)
    ref = sum(r for _, r in s)
    return sum(g for g, _ in s) / ref if ref > 0 else math.nan


def _interval(records: list, statistic, block: int) -> tuple[float, float]:
    samples = block_bootstrap(records, statistic, block, RESAMPLES, BOOT_SEED)
    return percentile_interval([x for x in samples if not math.isnan(x)])


def outcome_counts(s: Summary) -> str:
    """Every year accounted for: better, tied or worse on common support, or outside it."""
    gaps = [c.gap for _, c in s.years if c.kind == "gap"]
    better, worse = sum(g > 1e-9 for g in gaps), sum(g < -1e-9 for g in gaps)
    parts = [
        f"dominates {s.kinds.get('dominates', 0)}",
        f"better on common support {better}",
        f"tied {len(gaps) - better - worse}",
        f"worse on common support {worse}",
        f"dominated {s.kinds.get('dominated', 0)}",
        f"no overlap {s.kinds.get('no_overlap', 0)}",
    ]
    return ", ".join(parts) + f" (of {len(s.years)} years)"


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
    lines.append(f"- Years: {outcome_counts(s)}")
    if s.mean_gap is None:
        lines.append("- No year with common support.")
        return lines
    n = s.kinds.get("gap", 0)
    if s.gap_ci is None:
        interval = f"no interval: fewer than {MIN_YEARS_FOR_INTERVAL} years with common support"
    else:
        interval = f"block-2 95% interval {s.gap_ci[0]:+.2f} to {s.gap_ci[1]:+.2f}"
    lines.append(f"- **Mean gap, conditional on common support ({n} years): {s.mean_gap:+.2f} {unit}** ({interval})")
    if s.ratio is None:
        lines.append(f"- Ratio: n/a (mean reference below {SMALL_REFERENCE} {unit})")
    else:
        cis = "; ".join(f"{k} {fmt_pct(a)} to {fmt_pct(b)}" for k, (a, b) in s.ratio_ci.items())
        lines.append(
            f"- **Ratio of sums, conditional on common support: {fmt_pct(s.ratio)}** ({cis}); "
            f"rule category: {category(s.ratio)}"
        )
    cells = []
    for y, c in s.years:
        if c.kind != "gap":
            cells.append(f"{y}: {c.kind}")
        else:
            pct = "" if c.reference < SMALL_REFERENCE else f" ({fmt_pct(c.gap / c.reference)})"
            cells.append(f"{y}: {c.gap:+.2f}{pct}")
    lines.append("- Per year (gap" + (", % of reference" if s.ratio is not None else "") + "): " + "; ".join(cells))
    return lines


def window_results(rows: list[dict], pack: str) -> list[str]:
    """Experiment 2 on the four 5-year windows (with the used pack's module loss): one value each, no interval."""
    windows = [r for r in rows if r["unit"] == "window" and r["exp"] == "2" and r["farm"] == "base"]
    means = seed_means(windows, "unmet_critical_kwh")
    starts = sorted({r["year"] for r in windows})
    lines = []
    for pol in ("fc_z1", "fc_climatology", "fc_perfect"):
        cells, pcts = [], []
        for y, c in per_year(means, pack, starts, EXP2_FIXED, [pol]):
            if c.kind != "gap":
                cells.append(f"{y}-{(y + 4) % 100:02d}: {c.kind}")
                continue
            pct = c.gap / c.reference if c.reference >= SMALL_REFERENCE else None
            pcts += [pct] if pct is not None else []
            cells.append(f"{y}-{(y + 4) % 100:02d}: {c.gap:+.2f} kWh/yr ({fmt_pct(pct)})")
        span = f"range {fmt_pct(min(pcts))} to {fmt_pct(max(pcts))}" if pcts else "no percentages"
        lines.append(f"- {pol}: {'; '.join(cells)}; {span}")
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
        f"\n{len(years)} weather years ({years[0]}-{years[-1]}), each a separate 1-year run (consecutive years "
        f"of one climate record, not independent draws); seeds averaged within the year. Every year is "
        f"categorised; gaps and ratios are **conditional on common support** of the Pareto envelopes. "
        f"Intervals resample all years in order in circular blocks ({RESAMPLES} resamples). Positive gap = "
        f"challenger better. In annual runs the used pack is an *initially degraded pack* with no module loss; "
        f"the 5-year windows include the loss. Placeholder loads, one site.",
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
                ("fc_z1", "realistic forecast z = 1 (primary); old 5-year runs, base: +43% new / +67% used"),
                ("fc_climatology", "climatology; old 5-year runs, base: +27% new / +56% used"),
                ("fc_perfect", "perfect weather; old 5-year runs, base: +31% new / +59% used; the rule needs >= 40%"),
                ("fc_z0", "z = 0; old 5-year runs, base: +34% new / +61% used"),
                ("fc_z2", "z = 2; old 5-year runs, base: +66% new / +80% used"),
            ):
                res[pol] = summarise(per_year(means, pack, years, EXP2_FIXED, [pol]))
                L += report_block(f"{farm} sizing, {PACK[pack]}: {pol}", "kWh/yr", res[pol], orig)
            if farm == "base":
                label = "new pack" if pack == "new" else "used pack, module loss in years 2-4"
                L.append(f"\n**Base sizing, four 5-year windows ({label}); one value per window, no interval:**\n")
                L += window_results(rows, pack)
                wear = window_wear(rows, "2", "base", pack, ["fixed_35"], ["fc_z1"])
                L.append(f"\nWear of fc_z1 against fixed_35 over 5 years ({label}): {fmt_wear(wear)}")
                ok = (
                    res["fc_z1"].ratio is not None
                    and res["fc_z1"].ratio >= CONFIRM
                    and (res["fc_perfect"].ratio or 0) >= 0.40
                    and all(w <= 1.0 for w in wear.values())
                )
                verdicts.append(
                    f"R-44 rule, base, {PACK[pack]}: z = 1 {category(res['fc_z1'].ratio)} on common support "
                    f"[{outcome_counts(res['fc_z1'])}], perfect "
                    f"{fmt_pct(res['fc_perfect'].ratio)} (needs >= 40%), max window wear "
                    f"{max(wear.values()):+.2f} pp -> {'CONFIRMED' if ok else 'not confirmed by the letter'}"
                )

    L.append("\n## Experiment 2b (post hoc): perfect weather against climatology, frontier against frontier")
    means = seed_means(by_exp[("2b", "base")], "unmet_critical_kwh")
    for pack in ("new", "used"):
        s = summarise(
            per_year(means, pack, years, [f"clim_m{m}" for m in MARGIN_TAGS], [f"perfect_m{m}" for m in MARGIN_TAGS])
        )
        L += report_block(f"base, {PACK[pack]}: perfect vs climatology", "kWh/yr", s, "only about 5-15% (C-104)")

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
                    L += report_block(f"{PACK[pack]}, {room} room: {fam}", unit, s, orig)
                    if metric == "hours_above_max" and room == "produce" and fam == "multi":
                        wear = window_wear(rows, "3", "base", pack, ref, ch)
                        L.append(
                            f"\nWear of multi against batt_only over 5 years (windows, {pack} pack): {fmt_wear(wear)}"
                        )
                        verdicts.append(
                            f"R-45 rule, produce room, {PACK[pack]}: multi [{outcome_counts(s)}], "
                            f"max window wear {max(wear.values()):+.2f} pp"
                        )
            big = [f"bigbatt/batt_only/m{m:.2f}" for m in EXP3_MARGINS]
            bank = [f"bank/batt_only/m{m:.2f}" for m in EXP3_MARGINS]
            s = summarise(per_year(means, pack, years, big, bank))
            L += report_block(
                f"{PACK[pack]}, secondary: 6 kWh_th PCM bank vs a battery 3 kWh bigger (a trade-off: see the year "
                f"categories)",
                unit,
                s,
                "51-78% (C-108)",
            )

    L.append("\n## Verdicts by the original rules\n")
    L += [f"- {v}" for v in verdicts]
    L.append("\n## Reference families' realised unmet non-critical load (kWh/yr, seed means per year)\n")
    L += [f"- {name}: {min(xs):.1f} to {max(xs):.1f}" for name, xs in budgets]
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[-(len(verdicts) + len(budgets) + 4) :]))


if __name__ == "__main__":
    main()
