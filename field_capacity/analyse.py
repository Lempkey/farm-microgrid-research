"""Capacity trend, noise, test comparison and information droughts from natural episodes."""

import sys

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
S = sys.argv[1]
TESTS = {17: ["2019-05-03", "2020-06-27", "2021-09-25", "2022-03-13"], 19: ["2018-02-20", "2019-05-09"], 18: []}
NOM = {17: 158.0, 19: 200.0, 18: 158.0}


def theil_sen(x, y):
    i, j = np.triu_indices(len(x), 1)
    m = (y[j] - y[i]) / (x[j] - x[i])
    m = m[np.isfinite(m)]
    s = np.median(m)
    return s, np.median(y - s * x)


for sid in [17, 19, 18]:
    e = pl.read_csv(f"{S}/ep{sid}.csv").with_columns(pl.col("start").str.to_datetime())
    tdays = [np.datetime64(t) for t in TESTS[sid]]
    st = e["start"].to_numpy().astype("datetime64[D]")
    is_test = np.zeros(len(e), bool)
    for td in tdays:
        is_test |= np.abs((st - td).astype(int)) <= 1
    e = e.with_columns(pl.Series("test", is_test))
    nom = NOM[sid]
    print(f"\n================ ID {sid}  (nominal {nom} Ah) episodes={e.height}, near tests={is_test.sum()}")
    dis = e.filter((pl.col("kind") == "dis") & ~pl.col("test"))
    x = (dis["start"].to_numpy() - dis["start"].to_numpy()[0]).astype("timedelta64[m]").astype(float) / 525600
    y = dis["Ah"].to_numpy() / nom * 100
    if len(x) > 20:
        s, b = theil_sen(x, y)
        r = y - (b + s * x)
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        print(
            f"discharge episodes: n={len(y)}, Theil-Sen slope {s:.2f} pp/yr, start {b:.1f}% of nominal, residual sd(robust) {mad:.2f} pp, raw sd {r.std():.2f} pp"
        )
        # does the empty-anchor rest voltage explain the noise?
        v = dis["v_rest_empty"].to_numpy()
        T = dis["T"].to_numpy()
        A = np.column_stack([np.ones_like(x), x, v - v.mean(), T - np.nanmean(T)])
        ok = np.all(np.isfinite(A), 1)
        coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        r2 = y[ok] - A[ok] @ coef
        mad2 = 1.4826 * np.median(np.abs(r2 - np.median(r2)))
        print(
            f"  with rest-V and T terms: slope {coef[1]:.2f} pp/yr, dAh/dV_rest {coef[2]:.1f} pp per V (pack), dT {coef[3]:.2f} pp/K, residual sd(robust) {mad2:.2f} pp"
        )
        for sd, lab in [(mad, "raw"), (mad2, "corrected")]:
            n1 = (sd / 1.0) ** 2
            n05 = (sd / 0.5) ** 2
            print(f"  {lab}: episodes needed for +-1 pp (1 sigma): {n1:.1f}; for +-0.5 pp: {n05:.1f}")
    tst = e.filter(pl.col("test"))
    if tst.height:
        print("episodes on capacity-test dates:")
        print(tst.select("kind", "start", "hours", "Ah", "v_rest_empty", "T"))
    # monthly counts and medians
    m = (
        e.filter(~pl.col("test"))
        .group_by_dynamic("start", every="1mo")
        .agg(
            (pl.col("kind") == "dis").sum().alias("n_dis"),
            (pl.col("kind") == "chg").sum().alias("n_chg"),
            pl.col("Ah").filter(pl.col("kind") == "dis").median().round(1).alias("dis_med_Ah"),
        )
    )
    # days of data per month for normalisation
    d = pl.read_parquet(f"{S}/min/id{sid}.parquet").filter(pl.col("n") > 30)
    cov = (
        d.group_by_dynamic("Time", every="1mo").agg((pl.len() / 1440).round(1).alias("days")).rename({"Time": "start"})
    )
    m = cov.join(m, on="start", how="left").fill_null(0)
    m = m.with_columns(((pl.col("n_dis") + pl.col("n_chg")) / pl.col("days") * 30).round(1).alias("ep_per_30d"))
    print(m)
    # by calendar month, pooled
    cal = (
        m.filter(pl.col("days") >= 20)
        .with_columns(pl.col("start").dt.month().alias("mon"))
        .group_by("mon")
        .agg(pl.col("ep_per_30d").mean().round(1), pl.len().alias("months"))
        .sort("mon")
    )
    print("mean episodes per 30 days by calendar month (months with >=20 days of data):")
    print(cal.transpose(include_header=True))
    # gaps between consecutive episodes within continuous data
    ts = e["start"].to_numpy()
    gaps = np.diff(ts).astype("timedelta64[h]").astype(float) / 24
    p50, p90, p99, pmax = np.percentile(gaps, [50, 90, 99, 100])
    print(f"gap between episodes (days): median {p50:.1f}, p90 {p90:.1f}, p99 {p99:.1f}, max {pmax:.1f}")
    big = np.argsort(gaps)[-6:][::-1]
    print("largest gaps start:", [(str(ts[i])[:10], round(gaps[i], 1)) for i in big])
