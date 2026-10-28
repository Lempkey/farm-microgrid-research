# farm-microgrid-research

Open, early-stage research code on one question: **how can a small, low-cost solar-battery system keep
a rural farm's essential loads running through blackouts, for years?**

Two parts:

- **`src/farmsim/` + `experiments/`:** an hourly simulation of an off-grid farm microgrid.
  - The battery ages: calendar fade that doubles every 10 °C, plus depth-weighted cycle wear.
  - The controller learns the battery's capacity as it runs.
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
uv run pytest -q              # the model's behaviour, checkable by hand

# Weather: NASA POWER hourly, 2015-2024 (Valencia area, Venezuela, by default)
uv run python experiments/fetch_weather.py data/power

# Experiments use 4 worker processes by default; set MAX_WORKERS to change that
uv run python experiments/exp2_need_sized_reserve.py data/power 20 data/exp2_rows.json
uv run python experiments/summarise_exp2.py data/exp2_rows.json
```

A full sweep takes from minutes to about an hour, depending on the experiment and the core count.

## Experiments

Each experiment's decision rule was written down before its results were seen.

| Experiment | Question | Result so far (placeholder loads, one site) |
|---|---|---|
| 1. `exp1_capacity_knowledge.py` | Does knowing the battery's true capacity improve dispatch? Does deliberately probing it pay? | Barely: ≤ 2% change in unmet critical load. Probing costs more than it gains. |
| 2. `exp2_need_sized_reserve.py`, `exp2b_forecast_value.py` | Does a reserve sized to the coming days' need beat a fixed reserve? How much of that comes from the forecast? | A need-sized reserve beats the fixed-reserve frontier by 27–67%. Forecasts add 5–15% over climatology. |
| 3. `exp3_multi_store.py` | Does planning across battery, cold room and water tank beat battery-only protection? | Yes, mostly by pumping water with surplus PV, which is good practice already. Pre-cooling alone gives 14–31%. A thermal bank beats an equivalent battery by 51–78%. Absolute effects are small at this sizing. |

**Read the results with care:**

- **Comparisons are frontier against frontier.** Each policy is swept over safety margins and
  compared at equal unmet non-critical load.
- **Seeds are not independent.** Simulations start at `seed % 10` years into 10 years of weather and
  run 5 years, so 20 seeds cover only **10 overlapping weather windows**.
- **Weather is regional.** NASA POWER solar data sit on a 1° grid, so they are not a site
  measurement.

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
