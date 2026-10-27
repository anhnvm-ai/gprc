"""Check of the premise of the proposed serial-dependence stress test (27/10/2026, exploratory).

The paper attributes the gap between the empirical moving-block-bootstrap SE of beta_3 (0.124)
and its simulated counterpart to "more time clustering" in the data. This script compares the
observed panel with the baseline Gaussian design (cell 3: variant c, delta-hat, beta_3 = 0) on:
  * serial dependence: lag-1 autocorrelation of the monthly mean residual, of the residuals
    within each country and of the monthly score of beta_3; Bartlett(6) long-run variance ratio;
  * tails and concentration: excess kurtosis of GPRC (z) and of the beta_3 regressor after
    partialling out the other columns, and the share of its sum of squares in the top 5 months;
  * month bootstrap SE for block lengths 1, 3, 6 and 12, against the month-clustered SE.
Simulated panels: seeds of cell 3, replications 0-199 (block SEs: 0-59), regime estimated as in
the Monte Carlo study. No size or p-value of a new design is computed here.

Output: diag_round3_premise.json, in ../results (project layout) or next to this script (replication
repository, extra/). Usage: python <folder of this script>/diag_round3_premise.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "stage2").is_dir())
sys.path.insert(0, str(ROOT / "stage2" / "analysis"))
import a1_montecarlo as a1  # noqa: E402
import qflib as q  # noqa: E402

OUT = (HERE.parent / "results" if HERE.name == "scripts" else HERE) / "diag_round3_premise.json"


def ac1(z) -> float:
    z = np.asarray(z, float)
    return float(np.corrcoef(z[1:], z[:-1])[0, 1])


def measures(p: pd.DataFrame, blocks=()) -> dict:
    dz = q.design(p, q.STRESS)
    X, y, j, months = dz["X"], dz["y"], dz["pos"], dz["months"]
    _, resid, XtX_inv = q.ols_fit(X, y)
    others = np.delete(X, j, axis=1)
    xt = X[:, j] - others @ np.linalg.lstsq(others, X[:, j], rcond=None)[0]
    df = pd.DataFrame({"m": months, "c": dz["country"], "u": resid, "s": xt * resid, "x2": xt ** 2})
    score = df.groupby("m")["s"].sum().sort_index()
    z = score.values - score.values.mean()
    g0 = float(z @ z / len(z))
    lrv = g0 + 2 * sum((1 - L / 7) * float(z[L:] @ z[:-L] / len(z)) for L in range(1, 7))
    lev = df.groupby("m")["x2"].sum().sort_values(ascending=False)
    out = {"ac1_mean_resid": ac1(df.groupby("m")["u"].mean().sort_index()),
           "ac1_within_resid": float(np.mean([ac1(g.sort_values("m")["u"]) for _, g in df.groupby("c")])),
           "ac1_score": ac1(score), "lrv_ratio_bartlett6": lrv / g0,
           "kurt_regressor": float(pd.Series(xt).kurt()), "kurt_resid": float(pd.Series(resid).kurt()),
           "kurt_gprc": float(pd.Series(dz["frame"]["gpr_country_z"].values).kurt()),
           "top5_share": float(lev.head(5).sum() / lev.sum()),
           "top5_months": [str(t)[:7] for t in lev.head(5).index]}
    if blocks:
        out["se_cluster_month"] = float(q.se_cluster(X, resid, XtX_inv, months)[j])
        for b in blocks:
            out[f"se_block{b}"] = q.mbb_ols(X, y, months, j, B=999 if len(blocks) > 2 else 399, block=b, seed=42)[0]
    return out


def main() -> None:
    real = q.load_panel()
    res = {"data": measures(real, blocks=(1, 3, 6, 12))}
    res["data"]["kurt_gprc_by_country"] = (real.dropna(subset=["gpr_country_z"]).groupby("country")["gpr_country_z"]
                                           .apply(lambda s: float(s.kurt())).to_dict())
    par = a1.calibrate()["c"]
    sims = []
    for rep in range(200):
        rng = np.random.RandomState(a1.BASE_SEED + 30_000 + rep)
        y, g, s, f, e = a1.simulate_core(par, a1.T_OBS, a1.N_C, rng, par["dhat"], par["dhat"], 0.0)
        p, _ = a1.build_panel(y, g, s, f, e, rng)
        fm = q.fit_ms(q.regional_mean(p), starts=3)
        if not fm["ok"]:
            continue
        p[q.STRESS] = p.index.map(q.flag_threshold(fm["smoothed"]).to_dict())
        m = measures(p, blocks=(1, 6) if rep < 60 else ())
        m.pop("top5_months")
        sims.append(m)
    sd = pd.DataFrame(sims)
    sd["ratio_block1_cluster"] = sd["se_block1"] / sd["se_cluster_month"]
    sd["ratio_block6_cluster"] = sd["se_block6"] / sd["se_cluster_month"]
    d = res["data"]
    d["ratio_block1_cluster"] = d["se_block1"] / d["se_cluster_month"]
    d["ratio_block6_cluster"] = d["se_block6"] / d["se_cluster_month"]
    res["sim_cell3"] = {c: {"n": int(sd[c].notna().sum()), "q05": float(sd[c].quantile(.05)),
                            "median": float(sd[c].median()), "q95": float(sd[c].quantile(.95)),
                            "max": float(sd[c].max()), "share_ge_data": float((sd[c] >= d[c]).mean())}
                        for c in sd.columns if c in d}
    res["episodes_dropped_one_at_a_time"] = json.loads((ROOT / "stage2" / "results" / "qf_diag.json")
                                                       .read_text(encoding="utf-8"))["episodes"]
    res["_meta"] = {"date": "2026-10-27", "status": "exploratory", "seeds": "20261013 + 30000 + r, r = 0..199",
                    "note": "block SEs: data B = 999 (seed 42); simulations B = 399, replications 0-59"}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    keys = ["ac1_mean_resid", "ac1_score", "lrv_ratio_bartlett6", "kurt_gprc", "kurt_regressor", "top5_share",
            "ratio_block1_cluster", "ratio_block6_cluster"]
    for k in keys:
        s_ = res["sim_cell3"][k]
        print(f"{k:24s} data {d[k]:7.3f} | sim median {s_['median']:7.3f} [q05 {s_['q05']:.3f}, q95 {s_['q95']:.3f}], "
              f"share of sims >= data {s_['share_ge_data']:.2f}")
    print("data block SEs:", {b: round(d[f'se_block{b}'], 3) for b in (1, 3, 6, 12)}, "cluster:", round(d["se_cluster_month"], 3))
    print("top-5 months:", d["top5_months"], "| written", OUT.name)


if __name__ == "__main__":
    main()
