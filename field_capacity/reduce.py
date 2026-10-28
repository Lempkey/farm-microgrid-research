"""Reduce 1 s monthly CSVs to 1 min parquet: mean/min/max V, mean I, Ah, temperatures."""

import glob
import os
import sys

import polars as pl

src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{src}/*.csv"))
frames = []
for f in files:
    df = pl.read_csv(f, infer_schema_length=0, null_values=["NaN", ""])
    num = [c for c in df.columns if c != "Time"]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
    for c in ["V_in_V", "I_in_A", "T_Bat_in_C", "T_Room_in_C", "Interpolated"]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    df = df.with_columns(pl.col("Time").str.strptime(pl.Datetime, "%d-%b-%Y %H:%M:%S", strict=False))
    df = df.drop_nulls("Time").sort("Time")
    m = df.group_by_dynamic("Time", every="1m").agg(
        pl.col("V_in_V").mean().alias("V"),
        pl.col("V_in_V").min().alias("Vmin"),
        pl.col("V_in_V").max().alias("Vmax"),
        pl.col("I_in_A").mean().alias("I"),
        (pl.col("I_in_A").sum() / 3600).alias("Ah"),
        pl.col("I_in_A").abs().max().alias("Iabsmax"),
        pl.col("T_Bat_in_C").mean().alias("Tbat"),
        pl.col("T_Room_in_C").mean().alias("Troom"),
        pl.col("Interpolated").mean().alias("interp"),
        pl.len().alias("n"),
    )
    frames.append(m)
    print(os.path.basename(f), df.height, m.height, flush=True)
pl.concat(frames).sort("Time").write_parquet(out)
