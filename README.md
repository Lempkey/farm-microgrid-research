# farm-microgrid-research

Open, early-stage research code on one question: **how can a small, low-cost solar-battery system keep
a rural farm's essential loads running through blackouts, for years?**

Two parts:

- **`src/farmsim/` + `experiments/`:** an hourly simulation of an off-grid farm microgrid.
  - The battery ages: calendar fade that doubles every 10 °C, plus a cycle-wear proxy weighted by
    depth below full.
  - The controller learns the battery's capacity as it runs. Its filter holds a lone outlier and
    needs two consistent samples to accept a sudden capacity step.
  - The cold room is modelled as a thermal store.
  - Experiments compare reserve policies under real tropical weather.
- **`field_capacity/`:** estimating battery capacity from ordinary operation, with no dedicated tests.
  It works from public long-term field data on lithium iron phosphate (LFP) home batteries, and it
  also looks at an off-grid dataset from the Amazon.

**Status: preliminary.** Farm loads, battery packs and the cold room are *placeholders*, not
measurements of a real site. Treat results as directions to test, not findings. Issues and pull
requests are welcome.

## Quick start

```bash
uv sync --all-extras          # Python 3.12+
uv run pytest -q              # behaviour checkable by hand, energy conservation, regression snapshots

# Weather: NASA POWER hourly, 2001-2024 (Valencia area, Venezuela, by default)
uv run python experiments/fetch_weather.py data/power

# Experiments use 4 worker processes by default; set MAX_WORKERS to change that
uv run python experiments/exp2_need_sized_reserve.py data/power 20 data/exp2_rows.json
uv run python experiments/summarise_exp2.py data/exp2_rows.json

# Experiments 2, 2b and 3 per weather year, 2001-2024 (about 30 min on 4 workers)
uv run python experiments/per_year.py data/power data/per_year_rows.json
uv run python experiments/summarise_per_year.py data/per_year_rows.json results/per_year_2001_2024.md
```

One 5-year run takes about 0.05 s with a fixed reserve, 0.4 s with the need-sized reserve and 0.7 s
with the cold room (`experiments/benchmark.py`), so a full sweep takes seconds to minutes.

**Changing the model.** `tests/snapshots.json` pins six configurations at a tolerance. An intended
change regenerates it (`UPDATE_SNAPSHOTS=1 uv run pytest tests/test_snapshots.py`) and comes with a
delta table (`experiments/snapshot_delta.py <old> <new>`).

## Experiments

Each experiment's decision rule was written down before its results were seen.

| Experiment | Question | Result so far (placeholder loads, one site) |
|---|---|---|
| 1. `exp1_capacity_knowledge.py` | Does knowing the battery's true capacity improve dispatch? Does deliberately probing it pay? | Barely: ≤ 2% change in unmet critical load (1.9% at most, re-run after the October 2026 audit fixes; `gate_c100.py`). Probing costs more than it gains. |
| 2. `exp2_need_sized_reserve.py`, `exp2b_forecast_value.py` | Does a reserve sized to the coming days' need beat a fixed reserve? How much of that comes from the forecast? | Generally yes. With a realistic forecast it cuts unmet critical load by about 3–5 kWh/yr against the best fixed reserve at equal unmet other load: roughly 40% on common support (37%, interval 32–47%, new pack; 41%, 35–51%, initially degraded pack). It is better in 20 of 24 years for a new pack (the fixed reserve wins in 2, 2 are ties) and in all 24 for a degraded one. Over 5-year windows that include a module loss it gains 59–71% (14–22 kWh/yr). Perfect weather beats climatology by only 12–13%; the pre-registered bar for forecast value (≥ 40%) is not met. |
| 3. `exp3_multi_store.py` | Does planning across battery, cold room and water tank beat battery-only protection? | Yes, mostly by pumping water with surplus PV, which is good practice already: it dominates the battery-only frontier in 19 (new pack) and 22 (degraded pack) of 24 years and ties in the rest. Pre-cooling alone gives 17–19% fewer hours above 8 °C on common support. **A 6 kWh_th PCM bank against a battery 3 kWh larger is a trade-off:** in the 15 years where their frontiers overlap it cut overheating by roughly two-thirds (62–70% on common support; better in 10 and tied in 5 for a new pack, better in 14 and tied in 1 for a degraded one); in the other 9 neither cold room overheated and the larger battery served more household and irrigation load. Absolute effects are small at this sizing. |

**Read the results with care:**

- **Comparisons are frontier against frontier.** Each policy is swept over safety margins and
  compared at equal unmet non-critical load.
- **Experiments 2 and 3 are per weather year.** Each year of 2001–2024 is a separate 1-year run
  (10 seeds averaged within the year). The years are consecutive observations of one climate record,
  not independent draws, so intervals come from a circular block bootstrap that resamples all years in
  order. Every year is categorised (better, tied or worse on common support; dominates; dominated);
  percentages are **conditional on common support** (`results/per_year_2001_2024.md`).
- **"Initially degraded pack".** In annual runs the used pack starts degraded but never loses its
  module (switched off; in 5-year runs it happens in years 2–4). Wear and the module loss come from four
  non-overlapping 5-year windows, shown individually without intervals.
- **Experiment 1's seeds are not independent.** Its runs start on 1 January of year `seed % 10` of
  2015–2024 and run 5 calendar years, so 20 seeds cover only **10 overlapping weather windows**.
- **Weather is regional.** NASA POWER solar data sit on a 1° grid, so they are not a site
  measurement.

## Scope and known simplifications

This is a testbed with simple rule-based controllers. Its purpose is to find which levers matter
before committing to a method.

**There is no optimisation-based controller yet**, whether MPC, stochastic programming or
reinforcement learning. Those are the obvious next step, and the rule-based policies here are the
baselines they should beat.

**Known simplifications:**

- **Power system:**
  - The model is an hourly energy balance only.
  - It has no battery or inverter power limits, and no voltage or frequency.
- **Battery:**
  - The battery is a single energy store with a constant round-trip efficiency.
  - Calendar fade is linear in time, doubles every 10 °C, and does not depend on state of charge.
  - Cycle wear is a proxy, weighted by depth below full. There is no rainflow counting, so it
    tracks how deep the pack goes rather than each cycle's amplitude.
- **Forecasts are synthetic.** Each is the true hourly PV scaled by one lognormal error per lead
  day, so the hourly shape is always right.
- **The site is off-grid.** There is no grid connection and no model of outages; uncertainty comes
  from the weather alone.
- **Loads are placeholders.** Irrigation that isn't served is lost rather than deferred, and it
  carries no crop cost.
- **Cold room:** modelled as a single thermal node with a constant COP. The optional bank is an
  idealised phase-change material that holds exactly the 4 °C setpoint while it melts; ice is not
  modelled.

## Field capacity

The data are from Figgener et al., *Nature Energy* (2024), home storage systems in Germany:
[Zenodo 12091223](https://zenodo.org/records/12091223).

**Method:**
1. Each LFP system's 1-second logs are reduced to 1-minute data (`reduce.py`).
2. "Full" anchors are found at the top of charge, and "empty" anchors at rests on the steep low end
   of the LFP voltage curve (`anchors.py`).
3. The ampere-hours between consecutive opposite anchors are taken as one capacity sample.

**Results for system 17** (16 cells in series, 158 Ah):
- noise of about 1 percentage point per sample;
- 15–25 samples per 30 days from March to October, about 2 in midwinter;
- the estimates track the field capacity tests;
- one unexplained 6-point rise after the system was offline for a month.

```bash
# 1 s monthly CSVs (from Data_ID_17.zip) -> 1 min parquet
uv run python field_capacity/reduce.py <csv_dir> data/min/id17.parquet
# anchors: <parquet> <id> <v_full_pack> <v_empty_pack> <out>
uv run python field_capacity/anchors.py data/min/id17.parquet 17 56.6 49.6 data/ep17.csv
uv run python field_capacity/anchors.py data/min/id19.parquet 19 49.0 45.8 data/ep19.csv
uv run python field_capacity/anchors.py data/min/id18.parquet 18 55.8 51.5 data/ep18.csv
uv run python field_capacity/analyse.py data
```

`focus.py` also expects `ep19_44.9.csv` and `ep19_45.1.csv`, which come from the same command with
those empty thresholds. The `amazonas_*.py` scripts read Gaura et al.,
[Zenodo 3959739](https://zenodo.org/records/3959739). Their monitors read 100% state of charge almost all
the time and log almost no discharge current, so they cannot support capacity estimates.

## Data

No data is stored in this repository; the scripts download or read it. See each source for its
licence and how to cite it:

- **NASA POWER:** hourly, v2.10.
- **Figgener et al.:** Zenodo 12091223.
- **Gaura et al.:** Zenodo 3959739.

## Licence

MIT. See `LICENSE`.
