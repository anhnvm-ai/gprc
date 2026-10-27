"""Step A3 (part 2): numbers in the submitted paper whose
generating code was never saved, re-implemented with the specification of
ms3/ms7/ms9 (within-country z-scores, country dummies, normal p-values,
Bartlett Driscoll-Kraay with 4 lags).

  desc      Table 2 statistics and the correlations quoted in Sect. 4.1
  table4    OLS interaction with/without persistence: HC3, time-cluster, DK
  filtered  regime from filtered (not smoothed) MS probabilities
  nonlin    nonlinear / regime-specific persistence controls
  onset     ASEAN month-level onset logit (several candidate specifications;
            the one reproducing the submitted numbers is used for 'cr')
  perm      5,000 stress-label permutations (seed 42, procedure of ms9)

Input : shadow_<variant>/data/processed/asean_panel_regimes.parquet
Output: results/extra_<variant>.json (+ results/logs/cr3_extra_<variant>.md)
Each block is cached inside the json, so a rerun only computes missing blocks.

Usage: python scripts/cr3_extra.py raw|cr [--force]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crutil import CR, RES, LOGS, save_json, load_json  # noqa: E402

warnings.filterwarnings("ignore")
CC5 = ["VNM", "THA", "IDN", "MYS", "PHL"]
STRESS = "stress_common"
OUTCOME = "log_rv_lead1_z"
CTRL = ["vix_m_z", "dlog_fx_m_z", "sp500_ret_m_z", "us10y_m_z", "wti_ret_m_z", "ret_m_z"]
ZCOLS = ["gpr_country", "vix_m", "dlog_fx_m", "log_rv_t", "log_rv_lead1", "ret_lead1",
         "sp500_ret_m", "us10y_m", "wti_ret_m", "ret_m"]


# ----------------------------------------------------------------------------- data
def load_panel(variant: str) -> pd.DataFrame:
    path = CR / f"shadow_{variant}" / "data" / "processed" / "asean_panel_regimes.parquet"
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    p = p.set_index("date").sort_index()
    p = p[p["country"].isin(CC5)].copy()
    p["log_rv"] = np.log(p["rv_m"].clip(lower=1e-8))
    p["log_rv_lead1"] = p.groupby("country")["log_rv"].shift(-1)
    p["ret_lead1"] = p.groupby("country")["ret_m"].shift(-1)
    p["log_rv_t"] = p["log_rv"]
    for c in ZCOLS:
        p[c + "_z"] = p.groupby("country")[c].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12))
    return p


# ----------------------------------------------------------------------------- inference
def driscoll_kraay(X, y, time_idx, maxlags=4):
    """Identical to ms7_inference_robust.driscoll_kraay."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    h = X * resid[:, None]
    ht = pd.DataFrame(h, index=time_idx).groupby(level=0).sum().sort_index().values
    S = ht.T @ ht
    for lag in range(1, maxlags + 1):
        w = 1 - lag / (maxlags + 1)
        G = ht[lag:].T @ ht[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def interaction_fit(p: pd.DataFrame, stress: pd.Series | str, *, persistence=True,
                    extra: dict | None = None, outcome=OUTCOME, dk_lags=(4,)):
    """OLS of Eq. (3) with country dummies. `stress` is a column name or a
    date-indexed series (0/1 or probability). Returns b3 with several SEs."""
    d = p.copy()
    d["S"] = d[stress] if isinstance(stress, str) else d.index.map(stress.to_dict())
    d["GxS"] = d["gpr_country_z"] * d["S"]
    cols = ["gpr_country_z", "S", "GxS"] + (["log_rv_t_z"] if persistence else []) + CTRL
    for name, fn in (extra or {}).items():
        d[name] = fn(d)
        cols.append(name)
    d = d.dropna(subset=[outcome] + cols)
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    Xdf = pd.concat([d[cols].reset_index(drop=True), cty.reset_index(drop=True)], axis=1).astype(float)
    X = sm.add_constant(Xdf.values)
    y = d[outcome].values
    pos = 1 + cols.index("GxS")
    out = {"n": int(len(d)), "stress_obs": int((d["S"] > 0.5).sum())}
    m_hc3 = OLS(y, X).fit(cov_type="HC3")
    b = float(m_hc3.params[pos])
    out["b3"] = b
    if extra:
        out["extra_coefs"] = {nm: {"coef": float(m_hc3.params[1 + cols.index(nm)]),
                                   "p_hc3": float(m_hc3.pvalues[1 + cols.index(nm)])}
                              for nm in extra}

    def pack(se):
        z = b / se
        return {"se": float(se), "p": float(2 * (1 - norm.cdf(abs(z)))),
                "ci": [float(b - 1.96 * se), float(b + 1.96 * se)]}
    out["hc3"] = pack(m_hc3.bse[pos])
    tg = pd.factorize(d.index)[0]
    out["time_cluster"] = pack(OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": tg}).bse[pos])
    for L in dk_lags:
        _, se_dk = driscoll_kraay(X, y, d.index, maxlags=L)
        out[f"dk{L}"] = pack(se_dk[pos])
    return out


# ----------------------------------------------------------------------------- Markov
def fit_ms(y: pd.Series, n_restarts=10, seed=42):
    """Same estimation loop as src/models/markov_regime.fit_markov_2state, but
    returning smoothed AND filtered stress probabilities."""
    y = y.dropna()
    yv = np.asarray(y, dtype=float)
    best = None
    for rep in range(n_restarts):
        try:
            mod = MarkovRegression(yv, k_regimes=2, trend="c", switching_variance=False)
            if rep == 0:
                res = mod.fit(em_iter=40, maxiter=200)
            else:
                rng = np.random.RandomState(seed + rep)
                sp0 = mod.start_params * (1 + 0.15 * rng.randn(len(mod.start_params)))
                res = mod.fit(start_params=sp0, em_iter=40, maxiter=200)
            if np.isfinite(res.llf) and (best is None or res.llf > best.llf):
                best = res
        except Exception:
            continue
    smp = np.asarray(best.smoothed_marginal_probabilities)
    fmp = np.asarray(best.filtered_marginal_probabilities)
    assign = smp.argmax(axis=1)
    means = [yv[assign == k].mean() if (assign == k).any() else -np.inf for k in range(2)]
    s = int(np.argmax(means))
    return (pd.Series(smp[:, s], index=y.index), pd.Series(fmp[:, s], index=y.index))


# ----------------------------------------------------------------------------- blocks
def block_desc(p):
    """Table 2 uses the rows with an observed log RV (n=951); the Sect. 4.1
    correlations use the lead sample with within-country z-scores."""
    rows = p.dropna(subset=["log_rv"])
    d = p.dropna(subset=["log_rv_lead1", "log_rv_t"])

    def st(s):
        s = s.dropna()
        return {"n": int(len(s)), "mean": float(s.mean()), "std": float(s.std()),
                "min": float(s.min()), "median": float(s.median()), "max": float(s.max())}
    out = {"table2": {k: st(rows[c]) for k, c in [
        ("log_rv", "log_rv"), ("gprc", "gpr_country"), ("vix", "vix_m"),
        ("dlog_fx", "dlog_fx_m"), ("sp500", "sp500_ret_m"), ("domestic_ret", "ret_m")]}}
    out["corr_vix_rvlead_pooled_z"] = float(d["vix_m_z"].corr(d["log_rv_lead1_z"]))
    out["corr_vix_rvlead_by_country"] = {c: float(g["vix_m"].corr(g["log_rv_lead1"]))
                                         for c, g in d.groupby("country")}
    out["corr_vix_rvt_by_country"] = {c: float(g["vix_m"].corr(g["log_rv_t"]))
                                      for c, g in d.groupby("country")}
    out["corr_persistence_by_country"] = {c: float(g["log_rv_t"].corr(g["log_rv_lead1"]))
                                          for c, g in d.groupby("country")}
    out["corr_gprc_max_abs_z"] = float(max(abs(d["gpr_country_z"].corr(d[c]))
                                           for c in ["log_rv_t_z", "vix_m_z", "dlog_fx_m_z"]))
    return out


def block_table4(p):
    return {"no_persistence": interaction_fit(p, STRESS, persistence=False, dk_lags=(4, 6, 12)),
            "persistence": interaction_fit(p, STRESS, persistence=True, dk_lags=(4, 6, 12))}


def block_filtered(p):
    regional = p.groupby(p.index)["log_rv"].mean().dropna()
    smooth, filt = fit_ms(regional)
    sm_flag = (smooth > 0.5).astype(int)
    f_flag = (filt > 0.5).astype(int)
    stored = p.groupby(p.index)[STRESS].first().reindex(sm_flag.index)
    out = {"stress_months_smoothed": int(sm_flag.sum()),
           "smoothed_equals_stored": bool((stored.values == sm_flag.values).all()),
           "stress_months_filtered": int(f_flag.sum()),
           "agreement_share": float((sm_flag == f_flag).mean())}
    out["binary"] = interaction_fit(p, f_flag.astype(float))
    out["continuous"] = interaction_fit(p, filt)
    return out


def block_nonlin(p):
    sq = {"rv_sq": lambda d: d["log_rv_t_z"] ** 2}
    sq_x = {"rv_sq": lambda d: d["log_rv_t_z"] ** 2,
            "rv_x_stress": lambda d: d["log_rv_t_z"] * d["S"]}
    return {"plus_rv_sq": interaction_fit(p, STRESS, extra=sq),
            "plus_rv_sq_and_rv_x_stress": interaction_fit(p, STRESS, extra=sq_x)}


def _logit(y, X, names):
    Xc = sm.add_constant(X)
    m = sm.Logit(y, Xc).fit(disp=0)
    prob = m.predict(Xc)
    b, se = float(m.params[1]), float(m.bse[1])
    return {"coef": b, "se": se, "or": float(np.exp(b)),
            "or_ci": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
            "p": float(m.pvalues[1]), "auc": float(roc_auc_score(y, prob)),
            "pseudo_r2": float(m.prsquared), "brier": float(np.mean((prob - y) ** 2)),
            "regressors": names}


def block_onset(p):
    """ASEAN month-level onset logit. Specification recovered from the
    submitted numbers: GPRC = cross-country mean GPRC standardised over the
    calm-month sample; controls = month means of vix_m_z, sp500_ret_m_z and
    (raw) dlog_fx_m. On the raw panel this reproduces coef -0.041,
    SE 0.366, p 0.910, AUC 0.693, pseudo-R2 0.044."""
    m = p.groupby(p.index).mean(numeric_only=True)
    s = p.groupby(p.index)[STRESS].first()
    nxt = s.shift(-1)
    calm = (s == 0) & nxt.notna()
    y = nxt[calm].astype(int).values
    mm = m[calm]
    g = (mm["gpr_country"] - mm["gpr_country"].mean()) / mm["gpr_country"].std()
    ctrl = ["vix_m_z", "sp500_ret_m_z", "dlog_fx_m"]
    return {"calm_months": int(calm.sum()), "onsets": int(y.sum()),
            "gprc_only": _logit(y, g.values[:, None], ["gprc"]),
            "controlled": _logit(y, np.column_stack([g.values] + [mm[c].values for c in ctrl]),
                                 ["gprc"] + ctrl)}


def block_perm(p, n_perm=5000, seed=42):
    dates = p.index.unique()
    stress = p.groupby(p.index)[STRESS].first().reindex(dates)
    cols = ["gpr_country_z", "log_rv_t_z"] + CTRL
    d = p.dropna(subset=[OUTCOME] + cols).copy()
    code = pd.Index(dates).get_indexer(d.index)
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float).values
    base = d[cols].values.astype(float)
    y = d[OUTCOME].values
    g = d["gpr_country_z"].values

    def b3(svals):
        sp = svals[code]
        X = np.column_stack([np.ones(len(d)), sp, g * sp, base, cty])
        return np.linalg.lstsq(X, y, rcond=None)[0][2]
    true = b3(stress.values.astype(float))
    rng = np.random.RandomState(seed)
    vals = stress.values.astype(float).copy()
    draws = np.array([b3(rng.permutation(vals)) for _ in range(n_perm)])
    return {"b3_true": float(true), "n_perm": n_perm, "seed": seed,
            "p_one_sided": float(np.mean(draws <= true)),
            "p_two_sided": float(np.mean(np.abs(draws) >= abs(true))),
            "placebo_mean": float(draws.mean()), "placebo_sd": float(draws.std()),
            "draws_path": f"perm_draws_{{variant}}.npy", "_draws": draws}


BLOCKS = {"desc": block_desc, "table4": block_table4, "filtered": block_filtered,
          "nonlin": block_nonlin, "onset": block_onset, "perm": block_perm}


def main(variant: str, force: bool, redo: list[str] | None = None):
    out_path = RES / f"extra_{variant}.json"
    res = {} if force else load_json(out_path)
    for b in redo or []:
        res.pop(b, None)
    p = load_panel(variant)
    for name, fn in BLOCKS.items():
        if name in res:
            print(f"[{variant}] {name}: cached")
            continue
        print(f"[{variant}] {name}: computing ...", flush=True)
        r = fn(p)
        if name == "perm":
            np.save(RES / f"perm_draws_{variant}.npy", r.pop("_draws"))
            r["draws_path"] = f"perm_draws_{variant}.npy"
        res[name] = r
        save_json(res, out_path)          # checkpoint after every block
    print(f"[{variant}] saved {out_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["raw", "cr"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--redo", default="", help="comma-separated blocks to recompute")
    a = ap.parse_args()
    main(a.variant, a.force, [b for b in a.redo.split(",") if b])
