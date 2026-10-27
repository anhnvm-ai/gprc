"""A3 Flexible learners and richer controls (prespec P6).

  interaction : DML of GPRC x Stress for learners {lasso, ridge, rf, xgb, ensemble}
                x cross-fitting {shuffled, blocked} x control sets {C0 (paper), C1 (richer)}
  channels    : Table 3 channel ranking (persistence, VIX, FX, GPRC) for every learner
                x cross-fitting (C0); lasso reproduces the paper's Table 3
  alpha_path  : DML theta of GPRC x Stress for 12 fixed Lasso penalties (0.1x - 10x LassoCV)

C1 = C0 + 3- and 6-month average log RV, GPRC lags 1-2, global GPR, one-month lags of VIX,
dlog FX, WTI and US10Y (all z-scored within country; values missing because of lags are set
to the country mean, i.e. 0 in z units, so that N is unchanged).

Outputs: results/qf_a3.json (cached per block), results/tables/a3_*.csv, results/figures/a3_alpha_path.pdf
Usage: python stage2/analysis/a3_learners.py [--redo block]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402

OUT = q.RES / "qf_a3.json"
LEARNERS = ["lasso", "ridge", "rf", "xgb", "ensemble"]
SCHEMES = ["shuffled", "blocked"]
CHANNELS = {"persistence": "log_rv_t_z", "vix": "vix_m_z", "fx": "dlog_fx_m_z", "gprc": "gpr_country_z"}
MACRO = ["sp500_ret_m_z", "us10y_m_z", "wti_ret_m_z", "ret_m_z"]
C1_EXTRA = ["log_rv_ma3_z", "log_rv_ma6_z", "gprc_l1_z", "gprc_l2_z", "gpr_global_z",
            "vix_m_lag1_z", "dlog_fx_m_lag1_z", "wti_ret_m_lag1_z", "us10y_m_lag1_z"]


def add_c1(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = p.copy()
    g = p.groupby("country")
    p["log_rv_ma3"] = g["log_rv"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    p["log_rv_ma6"] = g["log_rv"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    p["gprc_l1"] = g["gpr_country"].shift(1)
    p["gprc_l2"] = g["gpr_country"].shift(2)
    imputed = {}
    for raw in ["log_rv_ma3", "log_rv_ma6", "gprc_l1", "gprc_l2", "gpr_global",
                "vix_m_lag1", "dlog_fx_m_lag1", "wti_ret_m_lag1", "us10y_m_lag1"]:
        z = q.zscore_within(p, raw)
        imputed[raw] = int(z.isna().sum())
        p[raw + "_z"] = z.fillna(0.0)
    return p, imputed


def block_interaction(p: pd.DataFrame) -> dict:
    p1, imputed = add_c1(p)
    out = {"imputed_missing": imputed}
    for cset in ["C0", "C1"]:
        d, ctrl = q.dml_frame(p1, q.STRESS)
        if cset == "C1":
            ctrl = ctrl + C1_EXTRA
        d = d.dropna(subset=[q.OUTCOME, "GxS"] + ctrl)
        Y, D, X, dates = d[q.OUTCOME].values, d["GxS"].values, d[ctrl].values.astype(float), d.index.values
        for ln in LEARNERS:
            for sch in SCHEMES:
                t0 = time.time()
                r = q.dml_theta(Y, D, X, dates, scheme=sch, model_y=q.learner(ln), model_d=q.learner(ln))
                out[f"{cset}|{ln}|{sch}"] = r
                print(f"  {cset} {ln:8s} {sch:8s} theta {r['theta']:+.3f} DK4 p {r['dk4']['p']:.3f} "
                      f"R2y {r['r2_y']:.2f} R2d {r['r2_d']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    return out


def block_by_regime(p: pd.DataFrame) -> dict:
    """ADDED AFTER SEEING THE 'interaction' BLOCK (not in the prespec; reported as such):
    with GPRC x Stress as the treatment and GPRC and Stress among the controls, RF/XGB predict
    the treatment almost exactly (out-of-fold R2_d 0.7-0.9), so that design is not identified for
    flexible learners. Here the treatment is GPRC itself, X contains Stress but not GPRC or
    GPRC x Stress, and delta = theta_stress - theta_calm (qflib.dml_by_regime)."""
    p1, _ = add_c1(p)
    cty = pd.get_dummies(p1["country"], prefix="cty", drop_first=True).astype(float)
    d0 = pd.concat([p1, cty], axis=1)
    base_ctrl = [q.STRESS, "log_rv_t_z"] + q.CTRL + list(cty.columns)
    out = {}
    for cset in ["C0", "C1"]:
        ctrl = base_ctrl + ([c for c in C1_EXTRA if c not in ("gprc_l1_z", "gprc_l2_z")] +
                            ["gprc_l1_z", "gprc_l2_z"] if cset == "C1" else [])
        d = d0.dropna(subset=[q.OUTCOME, "gpr_country_z"] + ctrl)
        args = (d[q.OUTCOME].values, d["gpr_country_z"].values, d[q.STRESS].values,
                d[ctrl].values.astype(float), d.index.values)
        for ln in LEARNERS:
            for sch in SCHEMES:
                r = q.dml_by_regime(*args, scheme=sch, model_y=q.learner(ln), model_g=q.learner(ln))
                out[f"{cset}|{ln}|{sch}"] = r
                print(f"  {cset} {ln:8s} {sch:8s} delta {r['delta']:+.3f} (DK4 p {r['dk4']['p']:.3f}) "
                      f"calm {r['theta_calm']:+.3f} stress {r['theta_stress']:+.3f} R2g {r['r2_g']:.2f}", flush=True)
    return out


def block_channels(p: pd.DataFrame) -> dict:
    cty = pd.get_dummies(p["country"], prefix="cty", drop_first=True).astype(float)
    d0 = pd.concat([p, cty], axis=1)
    out = {}
    for ch, col in CHANNELS.items():
        ctrl = [c for k, c in CHANNELS.items() if k != ch] + MACRO + list(cty.columns)
        d = d0.dropna(subset=[q.OUTCOME, col] + ctrl)
        for ln in LEARNERS:
            for sch in SCHEMES:
                r = q.dml_theta(d[q.OUTCOME].values, d[col].values, d[ctrl].values.astype(float),
                                d.index.values, scheme=sch, model_y=q.learner(ln), model_d=q.learner(ln),
                                dk_lags=(4,))
                out[f"{ch}|{ln}|{sch}"] = {k: r[k] for k in ["theta", "n", "iid", "dk4", "r2_y", "r2_d"]}
        print(f"  channel {ch}: " + ", ".join(f"{ln} {out[f'{ch}|{ln}|shuffled']['theta']:+.3f}"
                                              for ln in LEARNERS), flush=True)
    return out


def block_alpha_path(p: pd.DataFrame) -> dict:
    from src.models.dml import _make_model
    d, ctrl = q.dml_frame(p, q.STRESS)
    d = d.dropna(subset=[q.OUTCOME, "GxS"] + ctrl)
    Y, D, X, dates = d[q.OUTCOME].values, d["GxS"].values, d[ctrl].values.astype(float), d.index.values
    a_y = _make_model("lasso").fit(X, Y).named_steps["model"].alpha_
    a_d = _make_model("lasso").fit(X, D).named_steps["model"].alpha_
    mult = np.logspace(-1, 1, 12)
    out = {"alpha_y_cv": float(a_y), "alpha_d_cv": float(a_d), "multipliers": mult.tolist()}
    for sch in SCHEMES:
        res = [q.dml_theta(Y, D, X, dates, scheme=sch, model_y=q.lasso_fixed(m * a_y),
                           model_d=q.lasso_fixed(m * a_d), dk_lags=(4,)) for m in mult]
        out[sch] = [{"mult": float(m), "theta": r["theta"], "se_dk4": r["dk4"]["se"], "p_dk4": r["dk4"]["p"]}
                    for m, r in zip(mult, res)]
        print(f"  alpha path {sch}: theta from {min(x['theta'] for x in out[sch]):+.3f} "
              f"to {max(x['theta'] for x in out[sch]):+.3f}", flush=True)
    return out


def tables_and_figure(res: dict) -> None:
    tdir, fdir = q.RES / "tables", q.RES / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, r in res["interaction"].items():
        if "|" not in key:
            continue
        cset, ln, sch = key.split("|")
        rows.append({"controls": cset, "learner": ln, "crossfit": sch, "theta": round(r["theta"], 4),
                     "se_dk4": round(r["dk4"]["se"], 4), "p_dk4": round(r["dk4"]["p"], 4), "n": r["n"],
                     "r2_y": round(r["r2_y"], 3), "r2_d": round(r["r2_d"], 3)})
    pd.DataFrame(rows).to_csv(tdir / "a3_interaction_learners.csv", index=False)
    rows = []
    for key, r in res["channels"].items():
        ch, ln, sch = key.split("|")
        rows.append({"channel": ch, "learner": ln, "crossfit": sch, "theta": round(r["theta"], 4),
                     "p_dk4": round(r["dk4"]["p"], 4), "n": r["n"]})
    pd.DataFrame(rows).to_csv(tdir / "a3_channels_learners.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for sch, ls in [("shuffled", "-"), ("blocked", "--")]:
        path = res["alpha_path"][sch]
        m = [x["mult"] for x in path]
        th = np.array([x["theta"] for x in path])
        se = np.array([x["se_dk4"] for x in path])
        ax.plot(m, th, "k" + ls, marker="o", ms=3, label=f"{sch} cross-fitting")
        ax.fill_between(m, th - 1.96 * se, th + 1.96 * se, color="0.85" if sch == "shuffled" else "0.93", lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(1, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"Lasso penalty relative to the LassoCV choice", fontsize=9)
    ax.set_ylabel(r"$\hat\theta$ (GPRC $\times$ Stress)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fdir / "a3_alpha_path.pdf")
    fig.savefig(fdir / "a3_alpha_path.png", dpi=150)


BLOCKS = {"interaction": block_interaction, "channels": block_channels, "alpha_path": block_alpha_path,
          "by_regime": block_by_regime}


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
        res[name] = fn(p)
        q.save_json(res, OUT)                         # checkpoint after every block
    tables_and_figure(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", default="")
    main([b for b in ap.parse_args().redo.split(",") if b])
