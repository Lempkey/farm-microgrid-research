"""Daily and monthly battery behaviour for the four Amazonas off-grid houses (Gaura et al., Zenodo 3959739)."""

import glob
import sys

import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
src = sys.argv[1]
for h in "ABCD":
    fs = sorted(glob.glob(f"{src}/*STAR-{h}_Battery.csv"))
    d = pl.concat([pl.read_csv(f, infer_schema_length=0) for f in fs])
    d = (
        d.with_columns(
            pl.col("Time").str.to_datetime(strict=False),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in d.columns if c != "Time"],
        )
        .drop_nulls("Time")
        .sort("Time")
    )
    d = d.with_columns(
        (pl.col("SOC") / 10).alias("soc"), (pl.col("Voltage") / 1000).alias("V"), (pl.col("Current") / 1000).alias("I")
    )
    day = d.group_by_dynamic("Time", every="1d").agg(
        pl.col("soc").max().alias("smax"),
        pl.col("soc").min().alias("smin"),
        pl.col("V").max().alias("vmax"),
        pl.col("V").min().alias("vmin"),
        pl.len().alias("n"),
    )
    day = day.filter(pl.col("n") > 600)
    print(f"== STAR-{h}: {len(fs)} files, {d['Time'].min()} to {d['Time'].max()}, days with data {day.height}")
    print(
        "  V range",
        round(d["V"].quantile(0.01), 2),
        round(d["V"].median(), 2),
        round(d["V"].quantile(0.99), 2),
        " I range",
        round(d["I"].quantile(0.01), 1),
        round(d["I"].quantile(0.99), 1),
        " cycles",
        d["NumChargeCycles"].max(),
        " full discharges",
        d["NumFullDischarge"].max(),
    )
    m = day.group_by_dynamic("Time", every="1mo").agg(
        pl.len().alias("days"),
        (pl.col("smax") >= 99.9).mean().round(2).alias("frac_days_full"),
        (pl.col("smin") <= 30).mean().round(2).alias("frac_days_soc_le30"),
        pl.col("smin").median().alias("med_min_soc"),
        pl.col("vmax").median().round(2).alias("med_vmax"),
    )
    print(m)
