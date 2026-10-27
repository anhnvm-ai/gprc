"""A6 Main-table numbers after the decision of 13/10/2026:
  * main DML specification = interacted DML by regime (qflib.dml_by_regime), Lasso,
    shuffled and time-aware (blocked) cross-fitting;
  * main inference = wild bootstrap clustered by month (prespec P3, Monte Carlo);
    Driscoll-Kraay reported as secondary.

For every regime construction of the earlier-version Tables 4, 5 and A1 (plus the real-time
regime of A2), this script reports OLS (all inference procedures) and the by-regime DML.
Regime constructions follow stage1/scripts/cr4_experiments.py.

Output: results/qf_a6.json (one cached entry per regime construction)
Usage: python stage2/analysis/a6_main_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402

OUT = q.RES / "qf_a6.json"
B_WILD = 999


def best_fit(make_model, reps, seed0, sd):
    """cr4_experiments.best_fit: default start, then perturbed starts."""
    best = None
    for rep in range(reps):
        try:
            mod = make_model()
            if rep == 0 and seed0 is not None:
                res = mod.fit(em_iter=40, maxiter=200)
            else:
                rng = np.random.RandomState(seed0 + rep)
                sp0 = mod.start_params * (1 + sd * rng.randn(len(mod.start_params)))
                res = mod.fit(start_params=sp0, em_iter=40, maxiter=200)
            if np.isfinite(res.llf) and (best is None or res.llf > best.llf):
                best = res
        except Exception:
            continue
    return best


def byregime(d: pd.DataFrame, scol: str, scheme: str, persistence: bool = True, extra: list | None = None):
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    dd = pd.concat([d, cty], axis=1)
    ctrl = [scol] + (["log_rv_t_z"] if persistence else []) + q.CTRL + (extra or []) + list(cty.columns)
    dd = dd.dropna(subset=[q.OUTCOME, "gpr_country_z"] + ctrl)
    r = q.dml_by_regime(dd[q.OUTCOME].values, dd["gpr_country_z"].values, dd[scol].values,
                        dd[ctrl].values.astype(float), dd.index.values, scheme=scheme, wild_B=B_WILD)
    return {k: r[k] for k in ["delta", "theta_calm", "theta_stress", "dk4", "wild_month", "n", "r2_g"]}


def evaluate(d: pd.DataFrame, scol: str, persistence: bool = True, extra: dict | None = None) -> dict:
    dz = q.design(d, scol, persistence=persistence, extra=extra)
    o = q.ols_inference(dz, full=True, B_webb=B_WILD, B_month=B_WILD, B_mbb=499, seed=42)
    for name, fn in (extra or {}).items():
        d = d.assign(**{name: fn(d.assign(S=d[scol]))})
    out = {"stress_months": int(d.groupby(d.index)[scol].first().fillna(0).sum()) if d[scol].notna().any() else 0,
           "ols": o,
           "dml_shuffled": byregime(d, scol, "shuffled", persistence, list((extra or {}).keys())),
           "dml_blocked": byregime(d, scol, "blocked", persistence, list((extra or {}).keys()))}
    print(f"  OLS {o['b3']:+.3f} (wild-month p {o['wb_month']['p']:.3f}; DK p {o['dk4']['p']:.3f}) | "
          f"DML-reg shuffled {out['dml_shuffled']['delta']:+.3f} (wild p {out['dml_shuffled']['wild_month']['p']:.3f}) "
          f"| TA {out['dml_blocked']['delta']:+.3f} (wild p {out['dml_blocked']['wild_month']['p']:.3f})", flush=True)
    return out


def constructions(p: pd.DataFrame) -> dict:
    """name -> (frame, stress column, kwargs). Built lazily in main()."""
    reg = q.regional_mean(p)
    fm = q.fit_ms(reg, starts=10)
    smooth, filt = fm["smoothed"], fm["filtered"]
    q_ = p.copy()

    def put(name, flag):
        q_[name] = q_.index.map(pd.Series(flag).to_dict())
        return name

    cons = {}
    cons["baseline"] = (q_, q.STRESS, {})
    cons["baseline_no_persistence"] = (q_, q.STRESS, {"persistence": False})
    cons["filtered"] = (q_, put("S_filt", q.flag_threshold(filt)), {})
    q_["S_loco_top"] = q.loco_flags(p, starts=10, rule="top").values
    q_["S_loco_p05"] = q.loco_flags(p, starts=10, rule="p05").values
    cons["loco_top"] = (q_, "S_loco_top", {})
    cons["loco_p05"] = (q_, "S_loco_p05", {})
    lr = p.pivot_table(index=p.index, columns="country", values="log_rv")
    months = lr.dropna().index
    fb = q.fit_ms(q.regional_mean(p, months=months), starts=10)
    pb = p[p.index.isin(months)].copy()
    pb["S_bal_top"] = pb.index.map(q.flag_top_share(fb["smoothed"]).to_dict())
    pb["S_bal_p05"] = pb.index.map(q.flag_threshold(fb["smoothed"]).to_dict())
    cons["balanced_top"] = (pb, "S_bal_top", {})
    cons["balanced_p05"] = (pb, "S_bal_p05", {})
    vix = np.log(p.groupby(p.index)["vix_m"].first().dropna())
    cons["vix_q80"] = (q_, put("S_vix_q80", (vix > vix.quantile(0.80)).astype(float)), {})
    best = None
    for rep in range(30):                     # cr4_experiments.block_vix
        try:
            mod = MarkovRegression(vix.values, k_regimes=2, trend="c", switching_variance=True)
            if rep == 0:
                res = mod.fit(search_reps=20, search_iter=10, em_iter=40, maxiter=300)
            else:
                rng = np.random.RandomState(500 + rep)
                res = mod.fit(start_params=mod.start_params * (1 + 0.3 * rng.randn(len(mod.start_params))),
                              em_iter=40, maxiter=300)
            smp = np.asarray(res.smoothed_marginal_probabilities)
            share = (smp[:, 0] > 0.5).mean()
            if np.isfinite(res.llf) and 0.03 < share < 0.97 and (best is None or res.llf > best.llf):
                best = res
        except Exception:
            continue
    if best is not None:
        smp = np.asarray(best.smoothed_marginal_probabilities)
        a = smp.argmax(1)
        si = int(np.argmax([vix.values[a == k].mean() for k in range(2)]))
        cons["vix_ms"] = (q_, put("S_vix_ms", pd.Series((smp[:, si] > 0.5).astype(float), index=vix.index)), {})
    rt = pd.read_parquet(q.RES / "qf_a2_rt_probs.parquet")["p_rt"]
    cons["realtime_p05"] = (q_, put("S_rt", (rt > 0.5).astype(float)), {})
    # prespec P2 sensitivity: thresholds 0.4 and 0.6 (added 16/10/2026 for the P10 reporting rule)
    cons["realtime_p04"] = (q_, put("S_rt04", (rt > 0.4).astype(float)), {})
    cons["realtime_p06"] = (q_, put("S_rt06", (rt > 0.6).astype(float)), {})
    # additional controls (baseline regime)
    tot = q_.groupby(q_.index)["log_rv"].transform("sum")
    cnt = q_.groupby(q_.index)["log_rv"].transform("count")
    q_["rv_others"] = (tot - q_["log_rv"]) / (cnt - 1)
    q_["rv_others_z"] = q.zscore_within(q_, "rv_others")
    cons["plus_rv_sq"] = (q_, q.STRESS, {"extra": {"rv_sq": lambda d: d["log_rv_t_z"] ** 2}})
    cons["plus_rv_sq_rvxS"] = (q_, q.STRESS, {"extra": {"rv_sq": lambda d: d["log_rv_t_z"] ** 2,
                                                        "rv_x_stress": lambda d: d["log_rv_t_z"] * d["S"]}})
    cons["plus_spillover"] = (q_, q.STRESS, {"extra": {"rv_others_z": lambda d: d["rv_others_z"]}})
    # eight regime definitions of Table A1 (cr4_experiments.block_reg8)
    for thr, tag in [(0.4, "a_p04"), (0.5, "b_p05"), (0.6, "c_p06")]:
        cons[f"reg8_{tag}"] = (q_, put(f"S_{tag}", q.flag_threshold(smooth, thr)), {})
    b = best_fit(lambda: MarkovAutoregression(reg.values, k_regimes=2, order=1, switching_ar=True), 12, 200, 0.2)
    smp = np.asarray(b.smoothed_marginal_probabilities)
    idx = reg.index[1:] if smp.shape[0] == len(reg) - 1 else reg.index[:smp.shape[0]]
    assign = smp.argmax(1)
    yv = reg.values[1:][:len(assign)]
    si = int(np.argmax([yv[assign == k].mean() for k in range(2)]))
    cons["reg8_d_msar"] = (q_, put("S_msar", pd.Series((smp[:, si] > 0.5).astype(float), index=idx)), {})
    b3 = best_fit(lambda: MarkovRegression(reg.values, k_regimes=3, trend="c", switching_variance=False), 12, 300, 0.2)
    smp = np.asarray(b3.smoothed_marginal_probabilities)
    assign = smp.argmax(1)
    si = int(np.argmax([reg.values[assign == k].mean() for k in range(3)]))
    cons["reg8_e_3state"] = (q_, put("S_3st", pd.Series((assign == si).astype(float), index=reg.index)), {})
    for qq, tag in [(0.80, "f_q80"), (0.75, "g_q75")]:
        cons[f"reg8_{tag}"] = (q_, put(f"S_{tag}", (reg > reg.quantile(qq)).astype(float)), {})
    q_["S_cs"] = np.nan
    for c in q.CC5:
        sub = q_[q_["country"] == c]["log_rv"].dropna()
        fl = q.flag_threshold(q.fit_ms(sub, starts=10)["smoothed"])
        q_.loc[q_["country"] == c, "S_cs"] = q_.loc[q_["country"] == c].index.map(fl.to_dict())
    cons["reg8_h_country"] = (q_, "S_cs", {})
    return cons


def main() -> None:
    res = q.load_json(OUT)
    p = q.load_panel()
    cons = constructions(p)
    for name, (d, scol, kw) in cons.items():
        if name in res:
            print(f"{name}: cached")
            continue
        print(f"{name}:", flush=True)
        res[name] = evaluate(d, scol, persistence=kw.get("persistence", True), extra=kw.get("extra"))
        q.save_json(res, OUT)


if __name__ == "__main__":
    main()
