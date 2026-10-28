"""Find natural capacity-informative episodes in LFP home-storage field data.

Anchors: FULL = minute max voltage >= v_full (top of charge). EMPTY = a rest of >= 30 min
(|I| < 0.3 A) whose voltage after 30 min is <= v_empty (steep low-SoC end of the LFP OCV curve).
An episode is a pair of consecutive opposite anchors; its Ah is a capacity sample.
"""

import json
import sys

import numpy as np
import polars as pl

REST_MIN = 30
MAX_GAP_MIN = 10
MAX_EP_H = 72


def anchors(d, v_full, v_empty):
    V, Vmax, I, Iabs = (d[c].to_numpy() for c in ["V", "Vmax", "I", "Iabsmax"])
    t = d["Time"].to_numpy()
    full = Vmax >= v_full
    rest = (np.abs(I) < 0.3) & (Iabs < 1.0)
    ev = []
    i, N = 0, len(V)
    while i < N:
        if full[i]:
            j = i
            while j + 1 < N and (full[j + 1] or (j + 2 < N and full[j + 2])):
                j += 1
            ev.append(("F", i, j))
            i = j + 1
            continue
        if rest[i]:
            j = i
            while j + 1 < N and rest[j + 1]:
                j += 1
            if j - i + 1 >= REST_MIN and V[i + REST_MIN - 1] <= v_empty:
                ev.append(("E", i, j))
            i = j + 1
            continue
        i += 1
    return ev, t


def episodes(d, ev, t):
    Ah, Tb = d["Ah"].to_numpy(), d["Tbat"].to_numpy()
    V = d["V"].to_numpy()
    dt_min = np.diff(t).astype("timedelta64[m]").astype(int)
    out = []
    for (a, a0, a1), (b, b0, b1) in zip(ev, ev[1:], strict=False):
        if a == b:
            continue
        s, e = (a1 + 1, b0) if a == "F" else (a1 + 1, b0)
        if e <= s:
            continue
        if dt_min[s - 1 : e].max() > MAX_GAP_MIN:
            continue
        hours = (t[e] - t[s]).astype("timedelta64[m]").astype(int) / 60
        if hours > MAX_EP_H:
            continue
        q = Ah[s:e].sum()
        empty_idx = b0 if a == "F" else a0
        out.append(
            dict(
                kind="dis" if a == "F" else "chg",
                start=str(t[s])[:16],
                hours=round(hours, 1),
                Ah=round(abs(q), 2),
                v_rest_empty=round(float(V[empty_idx + REST_MIN - 1]), 3),
                T=round(float(np.nanmean(Tb[s:e])), 1),
            )
        )
    return pl.DataFrame(out)


if __name__ == "__main__":
    path, sid, v_full, v_empty, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    d = pl.read_parquet(path).filter(pl.col("n") > 30).sort("Time")
    ev, t = anchors(d, v_full, v_empty)
    ep = episodes(d, ev, t)
    ep.write_csv(out)
    nF = sum(1 for e in ev if e[0] == "F")
    nE = len(ev) - nF
    print(json.dumps(dict(id=sid, full_anchors=nF, empty_anchors=nE, episodes=ep.height)))
