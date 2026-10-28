"""System 17 in detail: trend fits before and after the 2022 gap, natural estimates around each field test,
and the step change after the system was offline in early 2022. Also system 19 at two empty thresholds."""

import sys

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
S = sys.argv[1]


def ts(x, y):
    i, j = np.triu_indices(len(x), 1)
    m = (y[j] - y[i]) / (x[j] - x[i])
    s = np.median(m[np.isfinite(m)])
    return s, np.median(y - s * x)


def fit(e, nom, lab):
    e = e.filter(pl.col("kind") == "dis")
    t0 = np.datetime64("2019-01-01")
    x = (e["start"].to_numpy().astype("datetime64[m]") - t0).astype(float) / 525600
    y = e["Ah"].to_numpy() / nom * 100
    v = e["v_rest_empty"].to_numpy()
    T = e["T"].to_numpy()
    s, b = ts(x, y)
    r = y - (b + s * x)
    sd = 1.4826 * np.median(np.abs(r - np.median(r)))
    A = np.column_stack([np.ones_like(x), x, v - v.mean(), T - np.nanmean(T)])
    ok = np.all(np.isfinite(A), 1)
    c, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    r2 = y[ok] - A[ok] @ c
    sd2 = 1.4826 * np.median(np.abs(r2 - np.median(r2)))
    print(
        f"{lab}: n={len(y)}  slope {s:.2f} pp/yr (TS), {c[1]:.2f} pp/yr (OLS+corr); robust sd raw {sd:.2f} pp, corrected {sd2:.2f} pp; rest-V coef {c[2]:.1f} pp/V; T coef {c[3]:.2f} pp/K"
    )
    return c


e17 = pl.read_csv(f"{S}/ep17.csv").with_columns(pl.col("start").str.to_datetime())
tests = ["2019-05-03", "2019-05-04", "2020-06-26", "2020-06-27", "2021-09-25", "2022-03-13", "2022-03-14"]
nat = e17.filter(~pl.col("start").dt.strftime("%Y-%m-%d").is_in(tests))
pre = nat.filter(pl.col("start") < pl.datetime(2022, 1, 1))
post = nat.filter(pl.col("start") >= pl.datetime(2022, 3, 1))
c = fit(pre, 158, "ID17 2019-03..2021-12")
fit(post, 158, "ID17 2022-03..2022-12")
# rolling 30-day natural estimate vs the test values
for td, q in [("2019-05-03", 144.29), ("2020-06-27", 144.35), ("2021-09-25", 140.87), ("2022-03-13", 152.75)]:
    t = np.datetime64(td)
    w = nat.filter(pl.col("kind") == "dis").filter(
        (pl.col("start") >= pl.lit(str(t - 30)).str.to_datetime()) & (pl.col("start") < pl.lit(td).str.to_datetime())
    )
    w2 = nat.filter(pl.col("kind") == "dis").filter(
        (pl.col("start") > pl.lit(td).str.to_datetime()) & (pl.col("start") <= pl.lit(str(t + 30)).str.to_datetime())
    )
    print(
        f"test {td}: test discharge {q} Ah | natural 30 d before: n={w.height} median {w['Ah'].median()} | 30 d after: n={w2.height} median {w2['Ah'].median()}"
    )
# step change check around Jan-Feb 2022
d = pl.read_parquet(f"{S}/min/id17.parquet").filter(
    (pl.col("Time") >= pl.datetime(2021, 12, 1)) & (pl.col("Time") < pl.datetime(2022, 3, 15)) & (pl.col("n") > 30)
)
dd = d.group_by_dynamic("Time", every="1d").agg(
    pl.col("Vmax").max().round(2), pl.col("Vmin").min().round(2), (pl.len() / 1440).round(2).alias("cov")
)
print(dd.filter((pl.col("Time") >= pl.datetime(2022, 1, 5)) & (pl.col("Time") < pl.datetime(2022, 3, 5))))
for ve in ["45.1", "44.9"]:
    e = pl.read_csv(f"{S}/ep19_{ve}.csv").with_columns(pl.col("start").str.to_datetime())
    good = e.filter(
        (pl.col("start") < pl.datetime(2016, 11, 1))
        | ((pl.col("start") >= pl.datetime(2017, 10, 1)) & (pl.col("start") < pl.datetime(2018, 4, 1)))
        | ((pl.col("start") >= pl.datetime(2018, 8, 1)) & (pl.col("start") < pl.datetime(2019, 1, 1)))
        | (pl.col("start") >= pl.datetime(2019, 8, 1))
    )
    print(
        f"ID19 v_empty {ve}: all episodes {e.height}; dis Ah quantiles",
        [round(e.filter(pl.col("kind") == "dis")["Ah"].quantile(q), 1) for q in [0.05, 0.25, 0.5, 0.75, 0.95]],
    )
    fit(good, 200, f"ID19 v_empty {ve}, well-covered months")
    m = (
        e.filter(pl.col("kind") == "dis")
        .group_by_dynamic("start", every="1mo")
        .agg(pl.len().alias("n"), pl.col("Ah").median().round(1).alias("med"), pl.col("T").mean().round(1))
    )
    print(m.transpose(include_header=True) if False else m)
