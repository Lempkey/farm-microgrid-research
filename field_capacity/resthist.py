"""Histogram of per-cell rest voltage after 30 minutes, to choose the "empty" anchor threshold."""

import sys

import numpy as np
import polars as pl

for sid, cells in [(17, 16), (19, 14), (18, 16)]:
    d = pl.read_parquet(f"{sys.argv[1]}/id{sid}.parquet").filter(pl.col("n") > 30).sort("Time")
    V, I, Ia = d["V"].to_numpy(), d["I"].to_numpy(), d["Iabsmax"].to_numpy()
    rest = (np.abs(I) < 0.3) & (Ia < 1.0)
    vs = []
    i = 0
    N = len(V)
    while i < N:
        if rest[i]:
            j = i
            while j + 1 < N and rest[j + 1]:
                j += 1
            if j - i + 1 >= 30:
                vs.append(V[i + 29] / cells)
            i = j + 1
        else:
            i += 1
    vs = np.array(vs)
    vs = vs[vs > 2.5]
    h, e = np.histogram(vs, bins=np.arange(2.80, 3.62, 0.04))
    print(f"ID {sid}: {len(vs)} rests >=30 min; per-cell V at 30 min")
    for k in range(len(h)):
        print(f"  {e[k]:.2f}-{e[k + 1]:.2f} {'#' * int(h[k] / max(h) * 60)} {h[k]}")
