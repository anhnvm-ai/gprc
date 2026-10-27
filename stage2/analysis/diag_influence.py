"""EXPLORATORY influence analysis fixed in the second addendum to the analysis plan
(docs/analysis_plan/addendum2_2026-10-27_influential_months.md in the replication repository).

  check   partial leverage q_t of every month (residual of the GPRC x Stress column on all other
          columns of the design, squared and summed by month, as in extra/diag_round3_premise.py);
          the five months with the largest q_t must be the set T of the addendum; the full-sample
          OLS run must reproduce beta_3 and its wild-bootstrap p-value in qf_a6.json
  I1, I2  OLS without each month of T and without all five; regime labels and z-scores are kept;
          restricted wild bootstrap clustered by month, Rademacher, B = 999, seed 42
  I3      time-aware DML by regime (Lasso) on the full sample and on the six samples of I1-I2
  I4      OLS without each month in turn (point estimates) and the delete-one-month jackknife SE
  I5      real-data regime-re-estimating bootstrap (settings of a1_montecarlo.py) with minimum
          stress-month counts of 3, 5 and 7; the run with 5 must reproduce mc/gr_real.json

Output: results/qf_influence.json (one cached entry per block), results/tables/influential_months.csv
Usage: python stage2/analysis/diag_influence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a6_main_tables as a6  # noqa: E402
import qflib as q  # noqa: E402

OUT = q.RES / "qf_influence.json"
CSV = q.RES / "tables" / "influential_months.csv"
B_WILD, SEED = 999, 42
# set T of the addendum, in its order: month -> partial leverage as printed there. The addendum prints
# 0.082 for 2010-05; the value is 0.08146 (0.0815 rounded again to 0.082 when the table was written).
# The check therefore requires the same months in the same order and each q within 0.001.
T_ADDENDUM = {"2026-03": 0.279, "2010-05": 0.082, "2025-04": 0.076, "2021-02": 0.053, "2010-04": 0.038}
MIN_STRESS = (3, 5, 7)


def ym(m) -> str:
    return str(pd.Timestamp(m))[:7]


def leverage(p: pd.DataFrame) -> pd.DataFrame:
    """Partial leverage of each month for beta_3, with the market that carries the largest share
    of the month and its GPRC (z). Uses the regressors only."""
    dz = q.design(p, q.STRESS)
    X, j = dz["X"], dz["pos"]
    others = np.delete(X, j, axis=1)
    xt = X[:, j] - others @ np.linalg.lstsq(others, X[:, j], rcond=None)[0]
    df = pd.DataFrame({"month": dz["months"], "market": dz["country"], "x2": xt ** 2,
                       "gprc_z": dz["frame"]["gpr_country_z"].values, "stress": dz["S"]})
    rows = []
    for m, g in df.groupby("month"):
        top = g.loc[g["x2"].idxmax()]
        rows.append({"month": pd.Timestamp(m), "q": g["x2"].sum(), "stress": int(g["stress"].iloc[0]),
                     "n_obs": len(g), "market_max": top["market"], "market_max_share": top["x2"] / g["x2"].sum(),
                     "market_max_gprc_z": top["gprc_z"]})
    lev = pd.DataFrame(rows).set_index("month")
    lev["q"] /= lev["q"].sum()
    lev = lev.sort_values("q", ascending=False)
    lev["rank"] = np.arange(1, len(lev) + 1)
    return lev


def ols_without(p: pd.DataFrame, drop, wild: bool = True) -> dict:
    """Eq. (1) by OLS without the months in `drop` (same code path as a6_main_tables.evaluate)."""
    d = p[~p.index.isin(list(drop))]
    dz = q.design(d, q.STRESS)
    X, y, j, months = dz["X"], dz["y"], dz["pos"], dz["months"]
    out = {"months_dropped": [ym(m) for m in drop], "n": int(len(y)), "T": int(len(np.unique(months))),
           "stress_months": int(len(np.unique(months[dz["S"] == 1]))), "b3": float(q.ols_fit(X, y)[0][j])}
    if wild:
        out["p_wild"] = q.wild_cluster_p(X, y, months, j, "rademacher", B_WILD, SEED)
    return out


def dml_without(p: pd.DataFrame, drop) -> dict:
    d = p[~p.index.isin(list(drop))]
    r = a6.byregime(d, q.STRESS, "blocked")
    return {"months_dropped": [ym(m) for m in drop], "n": r["n"], "delta": r["delta"],
            "p_wild": r["wild_month"]["p"], "theta_calm": r["theta_calm"], "theta_stress": r["theta_stress"]}


def block_check(p: pd.DataFrame, lev: pd.DataFrame) -> dict:
    top = {ym(m): float(v) for m, v in lev["q"].head(5).items()}
    if list(top) != list(T_ADDENDUM) or max(abs(top[k] - T_ADDENDUM[k]) for k in top) > 0.001:
        raise SystemExit(f"set T differs from the addendum: {top}")
    base = q.load_json(q.RES / "qf_a6.json")["baseline"]
    o = ols_without(p, [])
    same = o["b3"] == base["ols"]["b3"] and o["p_wild"] == base["ols"]["wb_month"]["p"]
    if not same:
        raise SystemExit(f"full-sample OLS does not reproduce qf_a6.json: {o}")
    return {"T": top, "top5_share": float(lev["q"].head(5).sum()), "n_months": int(len(lev)),
            "baseline_ols": o, "reproduces_qf_a6": same}


def block_drops(p: pd.DataFrame, T: list) -> dict:
    out = {ym(m): ols_without(p, [m]) for m in T}
    out["all5"] = ols_without(p, T)
    for k, v in out.items():
        print(f"  OLS without {k}: {v['b3']:+.3f} (p_wild {v['p_wild']:.3f}, n {v['n']})", flush=True)
    return out


def block_dml(p: pd.DataFrame, T: list) -> dict:
    base = q.load_json(q.RES / "qf_a6.json")["baseline"]["dml_blocked"]
    out = {"full": dml_without(p, [])}
    diff = out["full"]["delta"] - base["delta"]
    if round(out["full"]["delta"], 3) != round(base["delta"], 3):
        raise SystemExit(f"full-sample DML does not reproduce delta = {base['delta']:.3f}: {out['full']}")
    out["full"]["diff_vs_qf_a6"] = float(diff)
    for m in T:
        out[ym(m)] = dml_without(p, [m])
    out["all5"] = dml_without(p, T)
    for k, v in out.items():
        print(f"  TA-DML {k}: {v['delta']:+.3f} (p_wild {v['p_wild']:.3f}, n {v['n']})", flush=True)
    return out


def block_jackknife(p: pd.DataFrame, lev: pd.DataFrame, b_full: float) -> dict:
    b = pd.Series({m: ols_without(p, [m], wild=False)["b3"] for m in lev.index})
    G = len(b)
    se = float(np.sqrt((G - 1) / G * np.sum((b.values - b_full) ** 2)))
    return {"b3_without_month": {ym(m): float(v) for m, v in b.items()}, "jackknife_se": se,
            "min": float(b.min()), "argmin": ym(b.idxmin()), "max": float(b.max()), "argmax": ym(b.idxmax()),
            "n_negative": int((b < 0).sum()), "G": G}


def block_gr(p: pd.DataFrame) -> dict:
    """Copy of qflib.ms_regime_bootstrap with the settings of the real-data run in
    a1_montecarlo.py (B = 999, 6-month blocks, seed 42, 3 starts, smoothed P > 0.5). Every draw
    is recorded, so the minimum number of stress months can vary while the blocks stay the same."""
    B, block, starts = 999, 6, 3
    dz = q.design(p, q.STRESS)
    d = dz["frame"]
    months = np.sort(np.asarray(d.index.unique()))
    T = len(months)
    mpos = np.searchsorted(months, d.index.values)
    rows_list = [np.where(mpos == t)[0] for t in range(T)]
    reg = q.regional_mean(p).reindex(months)
    Xbase = dz["X"].copy()
    jS, jG, j = dz["names"].index("S"), dz["names"].index("gpr_country_z"), dz["pos"]
    rng = np.random.RandomState(SEED)
    nblocks = int(np.ceil(T / block))
    rec = []                                                         # (MS fit ok, stress months, beta_3)
    for _ in range(B):
        starts_ = rng.randint(0, T - block + 1, size=nblocks)
        seq = np.concatenate([np.arange(s, s + block) for s in starts_])[:T]
        fm = q.fit_ms(pd.Series(reg.values[seq]), starts=starts)
        if not fm["ok"]:
            rec.append((False, -1, np.nan))
            continue
        flag = (fm["smoothed"].values > 0.5).astype(float)
        k_s, b = int(flag.sum()), np.nan
        if 1 <= k_s <= T / 2:
            rows = np.concatenate([rows_list[t] for t in seq])
            sflag = np.concatenate([np.full(len(rows_list[t]), flag[k]) for k, t in enumerate(seq)])
            Xb = Xbase[rows].copy()
            Xb[:, jS] = sflag
            Xb[:, j] = Xb[:, jG] * sflag
            if len(np.unique(Xb[:, j])) >= 3:
                b = float(q.ols_coef(Xb, dz["y"][rows])[j])
        rec.append((True, k_s, b))
    ok = np.array([r[0] for r in rec])
    ns = np.array([r[1] for r in rec])
    bs = np.array([r[2] for r in rec])
    out = {"stress_months_per_draw": ns.tolist()}
    for m in MIN_STRESS:
        degen = ok & ((ns < m) | (ns > T / 2))
        keep = ok & ~degen & ~np.isnan(bs)
        dr = bs[keep]
        q25, q75 = np.percentile(dr, [25, 75])
        out[f"min{m}"] = {"se": float(dr.std(ddof=1)), "se_iqr": float((q75 - q25) / 1.349),
                          "p": float(2 * min((dr >= 0).mean(), (dr <= 0).mean())),
                          "B_used": int(keep.sum()), "degenerate_skipped": int(degen.sum())}
        print(f"  GR bootstrap, minimum {m} stress months: p {out[f'min{m}']['p']:.3f}, "
              f"used {out[f'min{m}']['B_used']}, skipped {out[f'min{m}']['degenerate_skipped']}", flush=True)
    real = q.load_json(q.RES / "mc" / "gr_real.json")
    r5 = out["min5"]
    out["reproduces_gr_real"] = bool(r5["p"] == real["p"] and r5["B_used"] == real["B_used"]
                                     and r5["degenerate_skipped"] == real["degenerate_skipped"]
                                     and np.isclose(r5["se"], real["se"], rtol=1e-12, atol=0))
    if not out["reproduces_gr_real"]:
        raise SystemExit(f"minimum 5 does not reproduce mc/gr_real.json: {r5} vs {real}")
    return out


def wording(r: float) -> str:
    if r >= 2 / 3:
        return "A: most of the estimate remains without the five months; abstract unchanged"
    if r >= 1 / 3:
        return "B: the estimate falls substantially but stays negative; one clause in the abstract"
    return "C: the association is concentrated in these months; abstract and conclusion reframed"


def main() -> None:
    res = q.load_json(OUT) if OUT.exists() else {}
    p = q.load_panel()
    lev = leverage(p)
    T = list(lev.index[:5])
    blocks = {"check": lambda: block_check(p, lev), "I1_I2": lambda: block_drops(p, T),
              "I3": lambda: block_dml(p, T)}
    for name, fn in blocks.items():
        if name in res:
            print(f"{name}: cached")
            continue
        print(f"{name}: computing", flush=True)
        res[name] = fn()
        q.save_json(res, OUT)
    if "I4" not in res:
        print("I4: computing", flush=True)
        res["I4"] = block_jackknife(p, lev, res["check"]["baseline_ols"]["b3"])
        q.save_json(res, OUT)
    if "I5" not in res:
        print("I5: computing (about 4 minutes)", flush=True)
        res["I5"] = block_gr(p)
        q.save_json(res, OUT)
    b0, b5 = res["check"]["baseline_ols"]["b3"], res["I1_I2"]["all5"]["b3"]
    res["decision"] = {"r": b5 / b0, "rule": "r >= 2/3: A; 1/3 <= r < 2/3: B; r < 1/3: C", "case": wording(b5 / b0),
                       "single_month_range": [min(v["b3"] for k, v in res["I1_I2"].items() if k != "all5"),
                                              max(v["b3"] for k, v in res["I1_I2"].items() if k != "all5")]}
    res["_meta"] = {"status": "exploratory", "addendum": "docs/analysis_plan/addendum2_2026-10-27_influential_months.md",
                    "wild_bootstrap": f"restricted, by month, Rademacher, B = {B_WILD}, seed {SEED}"}
    q.save_json(res, OUT)
    jk = res["I4"]["b3_without_month"]
    tab = lev.assign(month=[ym(m) for m in lev.index]).reset_index(drop=True)
    tab["b3_without_month"] = tab["month"].map(jk)
    tab["change_pct"] = 100 * (tab["b3_without_month"] - b0) / abs(b0)
    cols = ["month", "rank", "q", "stress", "n_obs", "market_max", "market_max_share", "market_max_gprc_z",
            "b3_without_month", "change_pct"]
    CSV.parent.mkdir(parents=True, exist_ok=True)
    tab[cols].round(6).to_csv(CSV, index=False)
    print(f"r = {b5 / b0:.3f} -> {res['decision']['case']}")
    print("written", OUT.name, "and", CSV.name)


if __name__ == "__main__":
    main()
