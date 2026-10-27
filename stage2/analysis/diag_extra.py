"""EXPLORATORY diagnostics added after the Monte Carlo size results (not in the prespec):

  episodes   leave-one-stress-episode-out: drop all months of one ex-post stress episode,
             re-estimate beta_3 (OLS: DK and wild bootstrap by month) and TA-DML
  dml_wild   wild bootstrap clustered by month (restricted, Rademacher, B = 999) for the
             final-stage DML regression eY on eD (nuisance residuals held fixed), shuffled and
             time-aware; plus the month-clustered (L = 0) DML score variance
  size_adj   size-adjusted p-value of the real-data DK statistic: share of H0 Monte Carlo
             replications (calibrated cells 3 and 5) whose DK p-value is <= the real one

Output: results/qf_diag.json
Usage: python stage2/analysis/diag_extra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402


def episodes(flag: pd.Series):
    out, start, prev = [], None, None
    for d, v in flag.items():
        if v == 1 and start is None:
            start = d
        if v != 1 and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return out


def block_episodes(p: pd.DataFrame) -> dict:
    st = p.groupby(p.index)[q.STRESS].first()
    out = {}
    for a, b in episodes(st):
        keep = ~((p.index >= a) & (p.index <= b))
        d = p[keep]
        dz = q.design(d, q.STRESS)
        o = q.ols_inference(dz, full=False)
        p_wb = q.wild_cluster_p(dz["X"], dz["y"], dz["months"], dz["pos"], "rademacher", 999, 42)
        r = q.dml_interaction(d, q.STRESS, scheme="blocked")
        key = f"{a.date()}..{b.date()}"
        out[key] = {"months_dropped": int(((st.index >= a) & (st.index <= b)).sum()), "ols_b3": o["b3"],
                    "ols_dk4_p": o["dk4"]["p"], "ols_wild_month_p": p_wb,
                    "tadml_theta": r["theta"], "tadml_dk4_p": r["dk4"]["p"]}
        print(f"  drop {key}: OLS {o['b3']:+.3f} (DK p {o['dk4']['p']:.3f}, wild-month p {p_wb:.3f}) "
              f"| TA-DML {r['theta']:+.3f} (p {r['dk4']['p']:.3f})", flush=True)
    return out


def dml_resid(p: pd.DataFrame, scheme: str):
    """Cross-fitted residuals of the paper's DML design (same folds as qflib.dml_theta)."""
    from src.models.dml import _make_model
    d, ctrl = q.dml_frame(p, q.STRESS)
    d = d.dropna(subset=[q.OUTCOME, "GxS"] + ctrl)
    Y, D, X, dates = d[q.OUTCOME].values, d["GxS"].values, d[ctrl].values.astype(float), d.index.values
    reps = 3 if scheme == "shuffled" else 1
    eYs, eDs = [], []
    for r in range(reps):
        sp = q.make_splits(dates, scheme, 5, seed=42 + 100 * r)
        eYs.append(q.cross_fit(Y, X, _make_model("lasso"), sp))
        eDs.append(q.cross_fit(D, X, _make_model("lasso"), sp))
    eY, eD = np.nanmean(eYs, axis=0), np.nanmean(eDs, axis=0)
    v = ~(np.isnan(eY) | np.isnan(eD))
    return eY[v], eD[v], dates[v]


def block_dml_wild(p: pd.DataFrame) -> dict:
    out = {}
    for sch in ["shuffled", "blocked"]:
        eY, eD, dv = dml_resid(p, sch)
        X = eD[:, None]
        theta = float(np.dot(eD, eY) / np.dot(eD, eD))
        p_wb = q.wild_cluster_p(X, eY, dv, 0, "rademacher", 999, 42)
        psi = eD * (eY - theta * eD)
        S = q.month_sums(psi[:, None], dv)
        se0 = float(np.sqrt(np.sum(S ** 2)) / np.sum(eD ** 2))
        out[sch] = {"theta_pooled_resid": theta, "wild_month_p": p_wb, "cluster_month_se": se0,
                    "cluster_month_p": q.pval(theta, se0), "n": int(len(eY))}
        print(f"  DML {sch}: theta {theta:+.3f}, wild-by-month p {p_wb:.3f}, month-cluster p "
              f"{q.pval(theta, se0):.3f}", flush=True)
    return out


def block_size_adj(p: pd.DataFrame) -> dict:
    tr = q.load_json(q.RES / "test_reproduce.json")
    ref = q.load_json(q.ROOT / "stage1" / "results" / "extra_cr.json")["table4"]["persistence"]
    p_dk_real = ref["dk4"]["p"]
    out = {"real_dk4_p": p_dk_real, "real_fixedb_p": tr["new_procedures_on_real_data"]["dk4_fixedb_p"]}
    for c in [0, 3, 5]:
        fs = sorted((q.RES / "mc" / f"cell{c}").glob("shard_*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in fs])
        out[f"cell{c}"] = {"n": int(len(df)),
                           "size_adjusted_dk4_p": float((df["ols_base_dk4_p"] <= p_dk_real).mean()),
                           "sd_b3_H0": float(df["ols_base_b3"].std()),
                           "mean_dk4_se_H0": float(df["ols_base_dk4_se"].mean()),
                           "mean_mbb_se_H0": float(df["ols_base_mbb_se"].mean())}
        print(f"  cell{c}: size-adjusted DK p {out[f'cell{c}']['size_adjusted_dk4_p']:.3f}; "
              f"sd(b3|H0) {out[f'cell{c}']['sd_b3_H0']:.3f} vs mean DK SE {out[f'cell{c}']['mean_dk4_se_H0']:.3f}",
              flush=True)
    dz = q.design(p, q.STRESS)
    beta, resid, XtX_inv = q.ols_fit(dz["X"], dz["y"])
    out["real_data_se"] = {"dk4": float(q.se_dk(dz["X"], resid, XtX_inv, dz["months"], 4)[dz["pos"]]),
                           "cl_month": float(q.se_cluster(dz["X"], resid, XtX_inv, dz["months"])[dz["pos"]]),
                           "mbb": 0.124}
    return out


BLOCKS = {"episodes": block_episodes, "dml_wild": block_dml_wild, "size_adj": block_size_adj}


def main() -> None:
    path = q.RES / "qf_diag.json"
    res = q.load_json(path)
    p = q.load_panel()
    for name, fn in BLOCKS.items():
        if name in res:
            print(f"{name}: cached")
            continue
        print(f"{name}: computing", flush=True)
        res[name] = fn(p)
        res[name + "_exploratory"] = True
        q.save_json(res, path)


if __name__ == "__main__":
    main()
