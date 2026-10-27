"""Tier 2 analyses (prespec P9), one cached block each:

  lp        B1 regime-dependent local projections, h = 0..6 (OLS-DK, L = max(4, h+1));
            time-aware DML at h = 1, 3, 6. h = 0 uses log RV_{t-1} as the persistence control.
  altvol    B2 alternative volatility measures from the same daily returns as RV: downside
            semivolatility, scaled mean absolute return, GARCH(1,1) volatility (MLE per market).
            Outcome and persistence control use the same measure; regime = baseline.
            (Range-based measures need High/Low data that are not in the project; not run.)
  tvtp      B3 time-varying transition probabilities: ASEAN-average GPRC (z, lag 1) in the
            transition equations of the 2-state MS; LR test (2 df), with and without VIX.
  speccurve B4 specification curve: the original ms12_spec_curve.py rerun on the corrected panel.
  econ      B5 economic value: volatility-managed portfolios and 5% VaR backtests from the A4
            forecasts (M2 vs M4), out-of-sample from 2020:01.

Output: results/qf_b.json, results/figures/b1_local_projections.pdf
Usage: python stage2/analysis/b_extras.py [--redo block1,block2]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402

OUT = q.RES / "qf_b.json"
DAILY = {"IDN": "asean_idn_daily", "THA": "asean_tha_daily", "MYS": "asean_mys_daily",
         "PHL": "asean_phl_daily", "VNM": "vnindex"}


# ----------------------------------------------------------------------------- B1
def block_lp(p: pd.DataFrame) -> dict:
    out = {}
    g = p.groupby("country")
    for h in range(0, 7):
        d = p.copy()
        if h == 0:
            d["y_h"] = d["log_rv"]
            d["log_rv_t_z"] = q.zscore_within(d.assign(l1=g["log_rv"].shift(1)), "l1")
        else:
            d["y_h"] = g["log_rv"].shift(-h)
        d["y_h_z"] = q.zscore_within(d, "y_h")
        dz = q.design(d, q.STRESS, outcome="y_h_z")
        beta, resid, XtX_inv = q.ols_fit(dz["X"], dz["y"])
        L = max(4, h + 1)
        se = q.se_dk(dz["X"], resid, XtX_inv, dz["months"], L)[dz["pos"]]
        b = float(beta[dz["pos"]])
        out[f"h{h}"] = {"b3": b, "se_dk": float(se), "p_dk": q.pval(b, se), "L": L, "n": int(len(dz["y"]))}
        if h in (1, 3, 6):
            dd, ctrl = q.dml_frame(d, q.STRESS)
            dd = dd.dropna(subset=["y_h_z", "GxS"] + ctrl)
            r = q.dml_theta(dd["y_h_z"].values, dd["GxS"].values, dd[ctrl].values.astype(float),
                            dd.index.values, scheme="blocked", dk_lags=(L,))
            out[f"h{h}"]["dml_blocked"] = {"theta": r["theta"], "se_dk": r[f"dk{L}"]["se"],
                                           "p_dk": r[f"dk{L}"]["p"], "n": r["n"]}
        print(f"  LP h={h}: b3 {b:+.3f} (DK{L} p {out[f'h{h}']['p_dk']:.3f})", flush=True)
    return out


def figure_lp(lp: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = list(range(0, 7))
    b = np.array([lp[f"h{h}"]["b3"] for h in hs])
    se = np.array([lp[f"h{h}"]["se_dk"] for h in hs])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.fill_between(hs, b - 1.96 * se, b + 1.96 * se, color="0.85", lw=0)
    ax.plot(hs, b, "k-o", ms=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Horizon h (months)", fontsize=9)
    ax.set_ylabel(r"$\hat\beta_3(h)$, 95% DK band", fontsize=9)
    fig.tight_layout()
    fdir = q.RES / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / "b1_local_projections.pdf")
    fig.savefig(fdir / "b1_local_projections.png", dpi=150)


# ----------------------------------------------------------------------------- B2
def daily_returns() -> dict[str, pd.Series]:
    out = {}
    for c, f in DAILY.items():
        d = pd.read_parquet(q.ROOT / "data" / "interim" / f"{f}.parquet")
        r = d["ret"] if "ret" in d else np.log(d["VNINDEX_close"]).diff()
        out[c] = r.dropna()
    return out


def garch11_vol(r: pd.Series) -> pd.Series:
    """Gaussian GARCH(1,1) by maximum likelihood; returns the conditional SD series."""
    x = (r - r.mean()).values
    v0 = x.var()

    def cond_var(th):
        w, a, b = th
        src = w + a * np.r_[v0, x[:-1] ** 2]
        h = lfilter([1.0], [1.0, -b], src, zi=[b * v0])[0]
        return np.maximum(h, 1e-12)

    def nll(th):
        w, a, b = th
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
            return 1e12
        h = cond_var(th)
        return 0.5 * np.sum(np.log(h) + x ** 2 / h)

    res = minimize(nll, x0=[0.05 * v0, 0.08, 0.90], method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
    return pd.Series(np.sqrt(cond_var(res.x)), index=r.index), res.x


def block_altvol(p: pd.DataFrame) -> dict:
    rets = daily_returns()
    rows, params = [], {}
    for c, r in rets.items():
        gv, th = garch11_vol(r)
        params[c] = {"omega": float(th[0]), "alpha": float(th[1]), "beta": float(th[2])}
        m = pd.DataFrame({"r": r, "garch": gv})
        grp = m.groupby(pd.Grouper(freq="MS"))
        agg = pd.DataFrame({
            "rv_check": grp["r"].std() * np.sqrt(252),
            "downside": np.sqrt(grp["r"].apply(lambda x: np.mean(np.where(x < 0, x ** 2, 0.0))) * 252),
            "mabs": grp["r"].apply(lambda x: np.mean(np.abs(x))) * np.sqrt(np.pi / 2) * np.sqrt(252),
            "garch": np.sqrt(grp["garch"].apply(lambda x: np.mean(x ** 2)) * 252)})
        agg["country"] = c
        rows.append(agg)
    mv = pd.concat(rows)
    mv.index.name = "date"
    out = {"garch_params": params}
    base = p.reset_index()
    for meas in ["rv_check", "downside", "mabs", "garch"]:
        d = base.merge(mv[[meas, "country"]].reset_index(), on=["date", "country"], how="left").set_index("date")
        d["m_t"] = np.log(d[meas].clip(lower=1e-8))
        d["m_lead"] = d.groupby("country")["m_t"].shift(-1)
        d["log_rv_t_z"] = q.zscore_within(d, "m_t")
        d["m_lead_z"] = q.zscore_within(d, "m_lead")
        dz = q.design(d, q.STRESS, outcome="m_lead_z")
        o = q.ols_inference(dz, full=False)
        dd, ctrl = q.dml_frame(d, q.STRESS)
        dd = dd.dropna(subset=["m_lead_z", "GxS"] + ctrl)
        r = q.dml_theta(dd["m_lead_z"].values, dd["GxS"].values, dd[ctrl].values.astype(float),
                        dd.index.values, scheme="blocked", dk_lags=(4,))
        out[meas] = {"ols_b3": o["b3"], "ols_p_dk4": o["dk4"]["p"], "n": o["n"],
                     "dml_blocked_theta": r["theta"], "dml_blocked_p_dk4": r["dk4"]["p"],
                     "corr_with_rv": float(np.corrcoef(np.log(d["rv_m"]), d["m_t"])[0, 1]) if meas != "rv_check" else 1.0}
        print(f"  {meas:9s} OLS b3 {o['b3']:+.3f} (p {o['dk4']['p']:.3f}) | TA-DML {r['theta']:+.3f} "
              f"(p {r['dk4']['p']:.3f})", flush=True)
    return out


# ----------------------------------------------------------------------------- B3
def _fit_mr(y, Z=None, reps=12, seed=7):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    best = None
    for rep in range(reps):
        try:
            mod = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=False, exog_tvtp=Z)
            if rep == 0:
                res = mod.fit(em_iter=40, maxiter=300) if Z is None else mod.fit(search_reps=30, maxiter=300)
            else:
                rng = np.random.RandomState(seed + rep)
                sp0 = mod.start_params * (1 + 0.2 * rng.randn(len(mod.start_params)))
                res = mod.fit(start_params=sp0, maxiter=300)
            if np.isfinite(res.llf) and (best is None or res.llf > best.llf):
                best = res
        except Exception:
            continue
    return best


def block_tvtp(p: pd.DataFrame) -> dict:
    reg = q.regional_mean(p)
    gm = p.groupby(p.index)["gpr_country"].mean().reindex(reg.index)
    vix = np.log(p.groupby(p.index)["vix_m"].first().reindex(reg.index))
    gz = ((gm - gm.mean()) / gm.std()).shift(1)
    vz = ((vix - vix.mean()) / vix.std()).shift(1)
    keep = gz.notna() & vz.notna()
    y = reg[keep].values
    one = np.ones(keep.sum())
    out = {}
    for tag, Z0, Z1 in [("gprc_only", one[:, None], np.column_stack([one, gz[keep]])),
                        ("gprc_with_vix", np.column_stack([one, vz[keep]]),
                         np.column_stack([one, vz[keep], gz[keep]]))]:
        r0 = _fit_mr(y, None if tag == "gprc_only" else Z0)
        r1 = _fit_mr(y, Z1)
        lr = 2 * (r1.llf - r0.llf)
        smp = np.asarray(r1.smoothed_marginal_probabilities)
        st = int(np.argmax([y[smp.argmax(1) == k].mean() if (smp.argmax(1) == k).any() else -np.inf
                            for k in range(2)]))
        names = list(r1.model.param_names)
        par = dict(zip(names, np.asarray(r1.params)))
        bse = dict(zip(names, np.asarray(r1.bse))) if hasattr(r1, "bse") else {}
        gcol = Z1.shape[1] - 1
        out[tag] = {"llf_restricted": float(r0.llf), "llf_tvtp": float(r1.llf), "LR": float(lr),
                    "df": 2, "p": float(chi2.sf(max(lr, 0.0), 2)), "stress_state": st,
                    "gprc_coefs": {k: {"coef": float(v), "se": float(bse.get(k, np.nan))}
                                   for k, v in par.items() if k.endswith(f"tvtp{gcol}")},
                    "n_months": int(keep.sum())}
        print(f"  TVTP {tag}: LR {lr:.2f} p {out[tag]['p']:.3f}", flush=True)
    return out


# ----------------------------------------------------------------------------- B4
def block_speccurve(p: pd.DataFrame) -> dict:
    shadow = q.ROOT / "stage1" / "rerun"
    res = subprocess.run([sys.executable, "scripts/ms12_spec_curve.py"], cwd=shadow,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    tab = shadow / "results" / "tables" / "tableMS12_spec_curve.csv"
    df = pd.read_csv(tab)
    bcol = [c for c in df.columns if c.lower() in ("b3", "coef", "beta3", "estimate")][0]
    pcol = [c for c in df.columns if c.lower().startswith("p")][0]
    out = {"returncode": res.returncode, "n_specs": int(len(df)), "coef_column": bcol, "p_column": pcol}
    # ms12 labels the lag-1 GPR specifications as timing placebos: summarise them separately
    for cls, g in df.groupby("spec_class"):
        out[cls] = {"n": int(len(g)), "n_negative": int((g[bcol] < 0).sum()),
                    "n_p_lt_05": int((g[pcol] < 0.05).sum()), "n_p_lt_10": int((g[pcol] < 0.10).sum()),
                    "median_coef": float(g[bcol].median()), "min_coef": float(g[bcol].min()),
                    "max_coef": float(g[bcol].max())}
        print(f"  spec curve [{cls}]: {len(g)} specs, negative {out[cls]['n_negative']}, "
              f"p<0.05 {out[cls]['n_p_lt_05']}, p<0.10 {out[cls]['n_p_lt_10']}", flush=True)
    return out


# ----------------------------------------------------------------------------- B5
def kupiec(hits: np.ndarray, a: float = 0.05) -> float:
    n, x = len(hits), int(hits.sum())
    pi = x / n
    if x in (0, n):
        ll1 = 0.0
    else:
        ll1 = x * np.log(pi) + (n - x) * np.log(1 - pi)
    ll0 = x * np.log(a) + (n - x) * np.log(1 - a)
    return float(chi2.sf(-2 * (ll0 - ll1), 1))


def christoffersen_lr(h: np.ndarray) -> float:
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(len(a), 1)

    def ll(pp, k0, k1):
        return (k0 * np.log(1 - pp) if k0 and pp < 1 else 0.0) + (k1 * np.log(pp) if k1 and pp > 0 else 0.0)
    return float(-2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11)))


def block_econ(p: pd.DataFrame) -> dict:
    import a4_forecast as a4
    pp = a4.prepare()
    f = a4.expanding_forecasts(pp, a4.NEW, "2020-01-01")
    nxt = p[["country", "ret_lead1"]].reset_index()
    f = f.merge(nxt, on=["date", "country"], how="left").dropna(subset=["ret_lead1"])
    out = {"n": int(len(f)), "note": "domestic index log returns (no risk-free rate available)"}
    gamma = 3.0
    var_lr_ind = {}
    for m in ["M0", "M2", "M4"]:
        sig2 = np.exp(2 * f[m]) / 12.0                    # monthly variance forecast
        sh, ce, hits_all = [], [], []
        lr_ind = 0.0
        for c, g in f.assign(sig2=sig2).groupby("country"):
            r = g["ret_lead1"].values
            w = 1.0 / g["sig2"].values
            rm = w * r
            rm = rm * (r.std() / rm.std())                 # same variance as buy-and-hold
            sh.append({"country": c, "sharpe_bh": r.mean() / r.std() * np.sqrt(12),
                       "sharpe_managed": rm.mean() / rm.std() * np.sqrt(12)})
            ce.append(12 * ((rm.mean() - gamma / 2 * rm.var()) - (r.mean() - gamma / 2 * r.var())))
            var = norm.ppf(0.95) * np.sqrt(g["sig2"].values)
            hits = (r < -var).astype(int)
            hits_all.append(hits)
            lr_ind += christoffersen_lr(hits)
        hits_all = np.concatenate(hits_all)
        var_lr_ind[m] = lr_ind
        out[m] = {"sharpe": sh, "mean_sharpe_managed": float(np.mean([s["sharpe_managed"] for s in sh])),
                  "mean_sharpe_bh": float(np.mean([s["sharpe_bh"] for s in sh])),
                  "mean_ce_gain_vs_bh_annual": float(np.mean(ce)),
                  "var5_hit_rate": float(hits_all.mean()), "kupiec_p": kupiec(hits_all),
                  "christoffersen_ind_p": float(chi2.sf(lr_ind, len(sh)))}
        print(f"  {m}: managed Sharpe {out[m]['mean_sharpe_managed']:.3f} (B&H {out[m]['mean_sharpe_bh']:.3f}), "
              f"VaR hits {out[m]['var5_hit_rate']:.3f} Kupiec p {out[m]['kupiec_p']:.3f}", flush=True)
    return out


BLOCKS = {"lp": block_lp, "altvol": block_altvol, "tvtp": block_tvtp, "speccurve": block_speccurve,
          "econ": block_econ}


def main(redo: list[str]) -> None:
    res = q.load_json(OUT)
    for b in redo:
        res.pop(b, None)
    p = q.load_panel()
    for name, fn in BLOCKS.items():
        if name in res:
            print(f"{name}: cached")
            continue
        print(f"{name}: computing", flush=True)
        try:
            res[name] = fn(p)
        except Exception as err:                       # record and continue with the other blocks
            res[name] = {"error": repr(err)}
            print(f"  {name} FAILED: {err!r}", flush=True)
        q.save_json(res, OUT)
    if "lp" in res and "error" not in res["lp"]:
        figure_lp(res["lp"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", default="")
    main([b for b in ap.parse_args().redo.split(",") if b])
