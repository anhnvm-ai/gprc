"""Phase B: additional checks of the GPRC x Stress
interaction (Eq. 3), all with the submitted specification (within-country
z-scores, country dummies, controls of ms3/ms7) and Driscoll-Kraay inference.

  dml_dk     shuffled and time-aware DML for GPRC x Stress, with the original
             i.i.d. score SE and a month-aggregated (Driscoll-Kraay-type) SE
  reg8       the eight alternative regime definitions of ms8 (Fig. 2), now
             reported with HC3, calendar-time-clustered and DK standard errors
  loco       leave-one-country-out regimes: country i is classified with a
             2-state MS on the average log RV of the other markets (OLS + DML)
  balanced   balanced five-market sample (all markets observed), MS re-fitted
  vix        externally defined regime: 2-state MS on monthly log VIX
  spillover  baseline regime + other-market average log RV at t as control

Input : shadow_<variant>/data/processed/asean_panel_regimes.parquet
Output: results/phaseB_<variant>.json (one cached block per check)

Usage: python scripts/cr4_experiments.py cr [--redo block1,block2]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crutil import CR, ROOT, RES, save_json, load_json  # noqa: E402
from cr3_extra import load_panel, interaction_fit, fit_ms, CTRL, STRESS, OUTCOME  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.models.dml import DMLEstimator, _cross_fit, make_blocked_splits  # noqa: E402

warnings.filterwarnings("ignore")
CC5 = ["VNM", "THA", "IDN", "MYS", "PHL"]
DK_LAGS = (4, 6, 12)


# ----------------------------------------------------------------------------- DML
def dml_dk(d: pd.DataFrame, treat: str, ctrl: list[str], scheme: str, seed=42):
    """Replicates DMLEstimator.fit (lasso, 5 folds; 3 reps if shuffled) and adds
    a Driscoll-Kraay-type SE: orthogonal scores are summed within each calendar
    month and a Bartlett HAC is applied to the monthly sums."""
    d = d.dropna(subset=[OUTCOME, treat] + ctrl)
    Y = d[OUTCOME].values.astype(float)
    D = d[treat].values.astype(float)
    X = d[ctrl].values.astype(float)
    dates = d.index.values
    est = DMLEstimator(ml_g="lasso", ml_m="lasso", n_folds=5,
                       n_reps=3 if scheme == "shuffled" else 1, seed=seed, cv_scheme=scheme)
    shuffle = scheme == "shuffled"
    splits = make_blocked_splits(dates, 5) if scheme == "blocked" else None
    reps = est.n_reps if shuffle else 1
    eY_all, eD_all, th = np.full((len(Y), reps), np.nan), np.full((len(Y), reps), np.nan), []
    for r in range(reps):
        s = est.seed + r * 100
        eY = _cross_fit(Y, X, est._ml_g_obj, 5, s, shuffle=shuffle, splits=splits)
        eD = _cross_fit(D, X, est._ml_m_obj, 5, s, shuffle=shuffle, splits=splits)
        v = ~(np.isnan(eY) | np.isnan(eD))
        th.append(np.dot(eD[v], eY[v]) / np.dot(eD[v], eD[v]))
        eY_all[:, r], eD_all[:, r] = eY, eD
    theta = float(np.median(th))
    eY, eD = np.nanmean(eY_all, axis=1), np.nanmean(eD_all, axis=1)
    v = ~(np.isnan(eY) | np.isnan(eD))
    eY, eD, dv = eY[v], eD[v], dates[v]
    psi = eD * (eY - theta * eD)
    J = np.sum(eD ** 2)
    out = {"theta": theta, "n": int(v.sum()), "scheme": scheme}
    se_iid = np.sqrt(np.mean(psi ** 2) / (len(psi) * np.mean(eD ** 2) ** 2))
    out["iid"] = {"se": float(se_iid), "p": float(2 * (1 - norm.cdf(abs(theta / se_iid))))}
    S = pd.Series(psi, index=dv).groupby(level=0).sum().sort_index().values
    for L in DK_LAGS:
        omega = np.sum(S ** 2)
        for lag in range(1, L + 1):
            omega += 2 * (1 - lag / (L + 1)) * np.sum(S[lag:] * S[:-lag])
        se = float(np.sqrt(omega) / J)
        out[f"dk{L}"] = {"se": se, "p": float(2 * (1 - norm.cdf(abs(theta / se)))),
                         "ci": [theta - 1.96 * se, theta + 1.96 * se]}
    return out


def dml_frame(p: pd.DataFrame, stress_col: str):
    d = p.copy()
    d["GxS"] = d["gpr_country_z"] * d[stress_col]
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    d = pd.concat([d, cty], axis=1)
    ctrl = ["gpr_country_z", stress_col, "log_rv_t_z"] + CTRL + list(cty.columns)
    return d, ctrl


# ----------------------------------------------------------------------------- helpers
def regional_mean(p: pd.DataFrame, exclude: str | None = None, months=None) -> pd.Series:
    q = p if exclude is None else p[p["country"] != exclude]
    r = q.groupby(q.index)["log_rv"].mean().dropna()
    return r if months is None else r[r.index.isin(months)]


def flag_from(smooth: pd.Series, thr=0.5) -> pd.Series:
    return (smooth > thr).astype(float)


def best_fit(make_model, reps, seed0, sd):
    """ms8-style multi-start: first fit default, then perturbed starts."""
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


def top_share(prob: pd.Series, k: int) -> pd.Series:
    """Stress = the k months with the highest stress probability (keeps the
    share of stress months equal to the baseline: k = 38 of 196)."""
    out = pd.Series(0.0, index=prob.index)
    out[prob.sort_values(ascending=False).index[:k]] = 1.0
    return out


BASE_SHARE = 38 / 196


def summarise(fit: dict) -> dict:
    keep = {k: fit[k] for k in ["b3", "n", "stress_obs"] + (["extra_coefs"] if "extra_coefs" in fit else [])}
    for s in ["hc3", "time_cluster"] + [f"dk{L}" for L in DK_LAGS if f"dk{L}" in fit]:
        keep[s] = fit[s]
    return keep


# ----------------------------------------------------------------------------- blocks
def block_dml_dk(p):
    d, ctrl = dml_frame(p, STRESS)
    return {s: dml_dk(d, "GxS", ctrl, s) for s in ["shuffled", "blocked"]}


def block_reg8(p):
    """Reproduces the eight constructions of ms8_regime_robust.py."""
    regional = regional_mean(p)
    smooth, _ = fit_ms(regional)
    defs = {}
    for thr in [0.4, 0.5, 0.6]:
        defs[f"MS mean, thr={thr}"] = ("Two-state switching-mean MS, smoothed P(stress) > %.1f" % thr,
                                       flag_from(smooth, thr))
    best = best_fit(lambda: MarkovAutoregression(regional.values, k_regimes=2, order=1,
                                                 switching_ar=True), 12, 200, 0.2)
    smp = np.asarray(best.smoothed_marginal_probabilities)
    idx = regional.index[1:] if smp.shape[0] == len(regional) - 1 else regional.index[:smp.shape[0]]
    assign = smp.argmax(1)
    yv = regional.values[1:][:len(assign)]
    si = int(np.argmax([yv[assign == k].mean() for k in range(2)]))
    defs["MarkovAR(1) switching-AR"] = ("Two-state MS-AR(1) with switching AR coefficient, P > 0.5",
                                        pd.Series((smp[:, si] > 0.5).astype(float), index=idx))
    best = best_fit(lambda: MarkovRegression(regional.values, k_regimes=3, trend="c",
                                             switching_variance=False), 12, 300, 0.2)
    smp = np.asarray(best.smoothed_marginal_probabilities)
    assign = smp.argmax(1)
    si = int(np.argmax([regional.values[assign == k].mean() for k in range(3)]))
    defs["3-regime MS (crisis)"] = ("Three-state MS; stress = highest-mean (crisis) state",
                                    pd.Series((assign == si).astype(float), index=regional.index))
    for q, lab in [(0.80, "Quantile top 20%"), (0.75, "Quantile top 25%")]:
        defs[lab] = (f"Regional log RV above its {int(q * 100)}th percentile",
                     (regional > regional.quantile(q)).astype(float))
    q = p.copy()
    q["stress_cs"] = np.nan
    for c in CC5:
        sub = q[q["country"] == c]["log_rv"].dropna()
        sm_c, _ = fit_ms(sub)
        q.loc[q["country"] == c, "stress_cs"] = q.loc[q["country"] == c].index.map(
            flag_from(sm_c).to_dict())
    out = {}
    for lab, (desc, s) in defs.items():
        out[lab] = {"definition": desc, **summarise(interaction_fit(p, s, dk_lags=DK_LAGS))}
    out["Country-specific MS"] = {"definition": "Two-state MS estimated separately for each market",
                                  **summarise(interaction_fit(q, "stress_cs", dk_lags=DK_LAGS))}
    return out


def block_loco(p):
    """Three LOCO constructions for market i, all without market i's own RV:
    ms      : 2-state MS on the other markets' average, P > 0.5
    ms_top  : same MS, stress = top 19.4% months by P (baseline share)
    q20     : other-market average log RV above its 80th percentile"""
    common = p.groupby(p.index)[STRESS].first()
    cols = {"ms": "stress_loco", "ms_top": "stress_loco_top", "q20": "stress_loco_q20"}
    q = p.copy()
    for c in cols.values():
        q[c] = np.nan
    info = {}
    for c in CC5:
        reg = regional_mean(p, exclude=c)
        sm_c, _ = fit_ms(reg)
        k = int(round(BASE_SHARE * len(reg)))
        series = {"ms": flag_from(sm_c), "ms_top": top_share(sm_c, k),
                  "q20": (reg > reg.quantile(0.80)).astype(float)}
        mask = q["country"] == c
        info[c] = {}
        for tag, fl in series.items():
            q.loc[mask, cols[tag]] = q.loc[mask].index.map(fl.to_dict())
            info[c][tag] = {"stress_months": int(fl.sum()),
                            "agreement_with_common": float((fl == common.reindex(fl.index)).mean())}
    out = {"models": info}
    for tag, col in cols.items():
        out[tag] = {"ols": summarise(interaction_fit(q, col, dk_lags=DK_LAGS))}
        d, ctrl = dml_frame(q, col)
        out[tag]["dml"] = {sch: dml_dk(d, "GxS", ctrl, sch) for sch in ["shuffled", "blocked"]}
    return out


def block_balanced(p):
    lr = p.pivot_table(index=p.index, columns="country", values="log_rv")
    months = lr.dropna().index                      # all five markets observed
    regional = regional_mean(p, months=months)
    sm_b, _ = fit_ms(regional)
    k = int(round(BASE_SHARE * len(regional)))
    common = p.groupby(p.index)[STRESS].first()
    q = p[p.index.isin(months)].copy()
    out = {"first_month": str(months.min().date()), "last_month": str(months.max().date()),
           "months": int(len(months))}
    for tag, fl in {"ms": flag_from(sm_b), "ms_top": top_share(sm_b, k),
                    "q20": (regional > regional.quantile(0.80)).astype(float),
                    "common_regime": common.reindex(months).astype(float)}.items():
        q["stress_bal"] = q.index.map(fl.to_dict())
        out[tag] = {"stress_months": int(fl.sum()),
                    "agreement_with_common": float((fl == common.reindex(fl.index)).mean()),
                    "ols": summarise(interaction_fit(q, "stress_bal", dk_lags=DK_LAGS))}
    return out


def block_vix(p):
    """External regime from monthly log VIX. The switching-mean MS degenerates
    (all months in one state), so we report (a) a switching-variance MS fitted
    with random-search starts and (b) VIX above its 80th percentile."""
    vix = np.log(p.groupby(p.index)["vix_m"].first().dropna())
    common = p.groupby(p.index)[STRESS].first()
    best = None
    for rep in range(30):
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
    out = {}
    if best is not None:
        smp = np.asarray(best.smoothed_marginal_probabilities)
        a = smp.argmax(1)
        si = int(np.argmax([vix.values[a == k].mean() for k in range(2)]))
        fl = pd.Series((smp[:, si] > 0.5).astype(float), index=vix.index)
        out["ms_sv"] = {"stress_months": int(fl.sum()),
                        "agreement_with_common": float((fl == common.reindex(fl.index)).mean()),
                        "ols": summarise(interaction_fit(p, fl, dk_lags=DK_LAGS))}
    fl = (vix > vix.quantile(0.80)).astype(float)
    out["q20"] = {"stress_months": int(fl.sum()),
                  "agreement_with_common": float((fl == common.reindex(fl.index)).mean()),
                  "ols": summarise(interaction_fit(p, fl, dk_lags=DK_LAGS))}
    return out


def block_spillover(p):
    q = p.copy()
    tot = q.groupby(q.index)["log_rv"].transform("sum")
    cnt = q.groupby(q.index)["log_rv"].transform("count")
    q["rv_others"] = (tot - q["log_rv"]) / (cnt - 1)
    q["rv_others_z"] = q.groupby("country")["rv_others"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))
    return {"ols": summarise(interaction_fit(q, STRESS, dk_lags=DK_LAGS,
                                             extra={"rv_others_z": lambda d: d["rv_others_z"]}))}


BLOCKS = {"dml_dk": block_dml_dk, "reg8": block_reg8, "loco": block_loco,
          "balanced": block_balanced, "vix": block_vix, "spillover": block_spillover}


def main(variant: str, redo: list[str]):
    out_path = RES / f"phaseB_{variant}.json"
    res = load_json(out_path)
    for b in redo:
        res.pop(b, None)
    p = load_panel(variant)
    for name, fn in BLOCKS.items():
        if name in res:
            print(f"[{variant}] {name}: cached")
            continue
        print(f"[{variant}] {name}: computing ...", flush=True)
        res[name] = fn(p)
        save_json(res, out_path)          # checkpoint after every block
    print(f"[{variant}] saved {out_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["raw", "cr"])
    ap.add_argument("--redo", default="")
    a = ap.parse_args()
    main(a.variant, [b for b in a.redo.split(",") if b])
