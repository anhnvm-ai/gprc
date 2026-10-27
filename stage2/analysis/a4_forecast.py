"""A4 Forecast evaluation (prespec P8).

Design (as the paper's ms5_regime_forecasting.py): one-step-ahead forecasts of log RV_{t+1} by
market; OLS re-estimated on all earlier rows of the same market (expanding window, >= 60
rows); unstandardised variables; out-of-sample rows from 2020:01 (robustness: from 2016:01).

Models (common out-of-sample rows for all models):
  M0 persistence          log RV_t
  M1 HAR                  + 3- and 6-month average log RV
  M2 HAR + controls       + VIX, dlog FX
  M3 + GPRC               + GPRC
  M4 + GPRC x P^RT        + GPRC, P^RT, GPRC x P^RT   (P^RT: real-time filtered stress
                          probability from A2 - no look-ahead)
Paper's specification (reproduction): log RV_t; + GPRC; + VIX; + FX; + all three.

Losses: squared error of log RV; QLIKE on the variance proxy RV^2 = exp(2 log RV) with the
forecast exp(2 yhat); squared error of RV = exp(log RV).
Metrics: Campbell-Thompson R2_OOS against the expanding historical mean (feasible benchmark);
R2 relative to M0; the paper's 'R2_OOS vs evaluation-sample mean' for reproduction.
Tests: Clark-West (one-sided, nested, vs M2 and vs M0) and Diebold-Mariano (two-sided) on the
month-averaged loss differentials with a Newey-West (4 lags) variance; Model Confidence Set
(Hansen, Lunde & Nason 2011, T_max statistic, moving-block bootstrap with blocks of 6 months,
1,000 draws, 90% set) on month-averaged losses. Conditional evaluation in calm vs stress
months, by the real-time regime (primary) and by the ex-post regime (the paper's choice).

Output: results/qf_a4.json, results/tables/a4_forecast.csv
Usage: python stage2/analysis/a4_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402

MIN_TRAIN = 60
NEW = {"M0": ["log_rv_t"], "M1": ["log_rv_t", "lrv_ma3", "lrv_ma6"],
       "M2": ["log_rv_t", "lrv_ma3", "lrv_ma6", "vix_m", "dlog_fx_m"],
       "M3": ["log_rv_t", "lrv_ma3", "lrv_ma6", "vix_m", "dlog_fx_m", "gpr_country"],
       "M4": ["log_rv_t", "lrv_ma3", "lrv_ma6", "vix_m", "dlog_fx_m", "gpr_country", "p_rt", "gprc_x_prt"]}
PAPER = {"P0_persistence": ["log_rv_t"], "P1_gprc": ["log_rv_t", "gpr_country"],
         "P2_vix": ["log_rv_t", "vix_m"], "P3_fx": ["log_rv_t", "dlog_fx_m"],
         "P4_full": ["log_rv_t", "gpr_country", "vix_m", "dlog_fx_m"]}


def prepare() -> pd.DataFrame:
    p = q.load_panel()
    rt = pd.read_parquet(q.RES / "qf_a2_rt_probs.parquet")["p_rt"]
    g = p.groupby("country")
    p["lrv_ma3"] = g["log_rv"].transform(lambda x: x.rolling(3, min_periods=3).mean())
    p["lrv_ma6"] = g["log_rv"].transform(lambda x: x.rolling(6, min_periods=6).mean())
    p["p_rt"] = p.index.map(rt.to_dict())
    p["gprc_x_prt"] = p["gpr_country"] * p["p_rt"]
    return p


def expanding_forecasts(p: pd.DataFrame, specs: dict, oos_start: str) -> pd.DataFrame:
    """One row per (market, origin month t >= oos_start) with forecasts of every model,
    the expanding historical mean and the realised outcome; rows common to all models."""
    feats_all = sorted({f for fs in specs.values() for f in fs})
    rows = []
    for c in q.CC5:
        d = p[p["country"] == c].dropna(subset=["log_rv_lead1"] + feats_all).sort_index()
        idx = np.where(d.index >= pd.Timestamp(oos_start))[0]
        for ti in idx:
            if ti < MIN_TRAIN:
                continue
            tr, row = d.iloc[:ti], d.iloc[ti]
            out = {"country": c, "date": d.index[ti], "y": row["log_rv_lead1"],
                   "hist_mean": tr["log_rv_lead1"].mean(), "stress_expost": row[q.STRESS],
                   "p_rt": row["p_rt"]}
            for m, fs in specs.items():
                Xtr = np.column_stack([np.ones(len(tr)), tr[fs].values])
                beta = np.linalg.lstsq(Xtr, tr["log_rv_lead1"].values, rcond=None)[0]
                out[m] = float(np.r_[1.0, row[fs].values] @ beta)
            rows.append(out)
    return pd.DataFrame(rows)


def nw_mean_test(x: pd.Series, L: int = 4) -> tuple[float, float]:
    """t-statistic of the mean of a monthly series with a Newey-West (Bartlett, L lags) variance."""
    v = x.values - x.values.mean()
    om = np.sum(v ** 2)
    for lag in range(1, L + 1):
        om += 2 * (1 - lag / (L + 1)) * np.sum(v[lag:] * v[:-lag])
    se = np.sqrt(om) / len(v)
    t = x.mean() / se if se > 0 else 0.0
    return float(t), float(se)


def losses(f: pd.DataFrame, m: str) -> pd.DataFrame:
    y, yh = f["y"], f[m]
    s2, h = np.exp(2 * y), np.exp(2 * yh)
    return pd.DataFrame({"mse_log": (y - yh) ** 2, "qlike": s2 / h - np.log(s2 / h) - 1,
                         "mse_rv": (np.exp(y) - np.exp(yh)) ** 2, "date": f["date"]})


def clark_west(f: pd.DataFrame, small: str, big: str) -> dict:
    """CW test of the nested pair (small within big); month-averaged adjusted differential."""
    e1, e2 = f["y"] - f[small], f["y"] - f[big]
    adj = e1 ** 2 - (e2 ** 2 - (f[small] - f[big]) ** 2)
    iid_t = adj.mean() / (adj.std(ddof=1) / np.sqrt(len(adj)))
    t, _ = nw_mean_test(adj.groupby(f["date"]).mean())
    return {"t_iid": float(iid_t), "p_iid": float(1 - norm.cdf(iid_t)),
            "t_nw_month": t, "p_nw_month": float(1 - norm.cdf(t))}


def diebold_mariano(f: pd.DataFrame, a: str, b: str, loss: str) -> dict:
    d = (losses(f, a)[loss] - losses(f, b)[loss]).groupby(f["date"]).mean()
    t, _ = nw_mean_test(d)
    return {"mean_diff": float(d.mean()), "t": t, "p": float(2 * (1 - norm.cdf(abs(t))))}


def mcs(Lmat: pd.DataFrame, alpha: float = 0.10, B: int = 1000, block: int = 6, seed: int = 42) -> dict:
    """Model Confidence Set, T_max statistic (Hansen, Lunde & Nason 2011), moving-block bootstrap.
    Lmat: T x m losses (rows = months). Returns MCS p-values and the 90% set."""
    rng = np.random.RandomState(seed)
    L = Lmat.values
    T, m = L.shape
    nb = int(np.ceil(T / block))
    idx = np.array([np.concatenate([np.arange(s, s + block) for s in rng.randint(0, T - block + 1, nb)])[:T]
                    for _ in range(B)])
    alive = list(range(m))
    pvals, running = {}, 0.0
    while len(alive) > 1:
        sub = L[:, alive]
        dbar_i = (sub - sub.mean(1, keepdims=True)).mean(0)               # d_i. = L_i - average
        boot = np.array([(sub[ix] - sub[ix].mean(1, keepdims=True)).mean(0) for ix in idx])
        var = ((boot - dbar_i) ** 2).mean(0)
        t_i = dbar_i / np.sqrt(var)
        tmax = t_i.max()
        tmax_star = ((boot - dbar_i) / np.sqrt(var)).max(1)
        p = float((tmax_star >= tmax).mean())
        running = max(running, p)
        worst = alive[int(np.argmax(t_i))]
        pvals[Lmat.columns[worst]] = running
        alive.remove(worst)
    pvals[Lmat.columns[alive[0]]] = 1.0
    return {"p_values": pvals, "set_90": [k for k, v in pvals.items() if v >= alpha]}


def evaluate(f: pd.DataFrame, specs: dict, bench: str, nested_vs: list[tuple[str, str]]) -> dict:
    out = {"n_forecasts": int(len(f)), "n_months": int(f["date"].nunique()),
           "first_target_origin": str(f["date"].min().date())}
    regimes = {"all": np.ones(len(f), bool),
               "calm_rt": (f["p_rt"] <= 0.5).values, "stress_rt": (f["p_rt"] > 0.5).values,
               "calm_expost": (f["stress_expost"] == 0).values, "stress_expost": (f["stress_expost"] == 1).values}
    for rname, sel in regimes.items():
        g = f[sel]
        if len(g) < 10:
            continue
        r = {"n": int(len(g))}
        sse_hist = np.sum((g["y"] - g["hist_mean"]) ** 2)
        sse_b = np.sum((g["y"] - g[bench]) ** 2)
        for m in specs:
            sse = np.sum((g["y"] - g[m]) ** 2)
            r[m] = {"r2_oos_ct": float(1 - sse / sse_hist), "r2_vs_bench": float(1 - sse / sse_b),
                    "r2_vs_eval_mean_paper": float(1 - sse / np.sum((g["y"] - g["y"].mean()) ** 2)),
                    "mean_qlike": float(losses(g, m)["qlike"].mean()),
                    "mean_mse_rv": float(losses(g, m)["mse_rv"].mean())}
        r["clark_west"] = {f"{b}>{s}": clark_west(g, s, b) for s, b in nested_vs}
        out[rname] = r
    out["diebold_mariano"] = {f"{a}-vs-{b}|{loss}": diebold_mariano(f, a, b, loss)
                              for a, b in [(bench, m) for m in specs if m != bench]
                              for loss in ["mse_log", "qlike", "mse_rv"]}   # prespec P8: three losses
    for loss in ["mse_log", "qlike", "mse_rv"]:
        Lm = pd.DataFrame({m: losses(f, m)[loss].values for m in specs}, index=f["date"]).groupby(level=0).mean()
        out[f"mcs_{loss}"] = mcs(Lm)
    return out


def main() -> None:
    p = prepare()
    res = {}
    for start in ["2020-01-01", "2016-01-01"]:
        f_new = expanding_forecasts(p, NEW, start)
        res[f"new_models_from_{start[:7]}"] = evaluate(
            f_new, NEW, "M0", [("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M2", "M4"), ("M3", "M4")])
    f_pap = expanding_forecasts(p, PAPER, "2020-01-01")
    res["paper_models_from_2020-01"] = evaluate(
        f_pap, PAPER, "P0_persistence",
        [("P0_persistence", m) for m in PAPER if m != "P0_persistence"])
    q.save_json(res, q.RES / "qf_a4.json")
    rows = []
    for key, blk in res.items():
        for rname in ["all", "calm_rt", "stress_rt", "calm_expost", "stress_expost"]:
            if rname not in blk:
                continue
            for m, v in blk[rname].items():
                if isinstance(v, dict) and "r2_oos_ct" in v:
                    rows.append({"design": key, "regime": rname, "model": m, "n": blk[rname]["n"],
                                 **{k: round(x, 4) for k, x in v.items()}})
    (q.RES / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(q.RES / "tables" / "a4_forecast.csv", index=False)
    b = res["new_models_from_2020-01"]
    print("OOS from 2020:01:", b["n_forecasts"], "forecasts")
    for rname in ["all", "calm_rt", "stress_rt"]:
        print(f"  {rname:10s}", {m: round(b[rname][m]["r2_oos_ct"], 3) for m in NEW},
              "| CW M4>M2 p", round(b[rname]["clark_west"]["M4>M2"]["p_nw_month"], 3),
              "| CW M3>M2 p", round(b[rname]["clark_west"]["M3>M2"]["p_nw_month"], 3))
    print("  MCS(mse_log)", b["mcs_mse_log"]["set_90"], "| MCS(qlike)", b["mcs_qlike"]["set_90"])
    pp = res["paper_models_from_2020-01"]
    print("paper design, R2 vs evaluation-sample mean (paper's 0.37 calm / -0.20 stress):",
          round(pp["calm_expost"]["P0_persistence"]["r2_vs_eval_mean_paper"], 3),
          round(pp["stress_expost"]["P0_persistence"]["r2_vs_eval_mean_paper"], 3))


if __name__ == "__main__":
    main()
