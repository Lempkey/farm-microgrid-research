"""Whole-record summary for the Amazonas houses: SoC resolution, current sign, voltage floors, Ah counter."""

import glob
import sys

import polars as pl

src = sys.argv[1]
for h in "ABCD":
    fs = sorted(glob.glob(f"{src}/*STAR-{h}_Battery.csv"))
    d = pl.concat([pl.read_csv(f, infer_schema_length=0) for f in fs])
    d = d.with_columns(
        pl.col("Time").str.to_datetime(strict=False),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in d.columns if c != "Time"],
    ).drop_nulls("Time")
    d = d.filter(pl.col("Time").dt.year() >= 2018)
    V = d["Voltage"] / 1000
    soc = d["SOC"] / 10
    I = d["Current"] / 1000
    n = d.height
    print(
        f"STAR-{h}: rows {n}; SOC unique values {soc.n_unique()}, min {soc.min()}, share at 100% {(soc >= 99.9).mean():.3f}; "
        f"I min {I.min():.1f} max {I.max():.1f}, share I<0 {(I < 0).mean():.3f}; share V<23.0 V {(V < 23.0).mean():.3f}, V<22.0 {(V < 22.0).mean():.3f}, V<21.0 {(V < 21.0).mean():.3f}; "
        f"Ah counter min {d['AmpereHours'].min() / 1000:.1f} max {d['AmpereHours'].max() / 1000:.1f}"
    )
