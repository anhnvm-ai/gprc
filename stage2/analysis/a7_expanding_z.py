"""A7 Expanding-window standardisation (27/10/2026). EXPLORATORY: added after
the analysis plan, as a check on the real-time exercise.

In the main analysis every regression variable is standardised within country with
full-sample moments (qflib.load_panel). Here each variable is standardised with the mean and
standard deviation of the same country's observations up to the current month only:
  * regressors dated t (GPRC, VIX, dlog FX, the controls): moments over months <= t;
  * log RV_t and the outcome log RV_{t+1}: moments of log RV over months <= t, so that the
    outcome scale also uses only information available at t;
  * at least MIN_HIST = 24 months of history per country; earlier rows are dropped.

Designs, each estimated with a6_main_tables.evaluate (OLS with all inference procedures, and
the DML by regime with Lasso, shuffled and time-aware cross-fitting, wild bootstrap by month):
  expost_fullz_restricted    ex-post regime, full-sample moments, restricted sample
  expost_expz                ex-post regime, expanding-window moments
  realtime_fullz_restricted  real-time regime (A2, filtered P > 0.5), full-sample moments,
                             restricted sample
  realtime_expz              real-time regime, expanding-window moments
The full-sample rows use the same observations as the expanding-window rows, so the two differ
only in the standardisation.

Output: results/qf_a7.json (one cached entry per design)
Usage: python stage2/analysis/a7_expanding_z.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402
import a6_main_tables as a6  # noqa: E402

OUT = q.RES / "qf_a7.json"
MIN_HIST = 24
REGRESSORS = ["gpr_country", "vix_m", "dlog_fx_m", "sp500_ret_m", "us10y_m", "wti_ret_m", "ret_m"]
USED = [q.OUTCOME, "gpr_country_z", "log_rv_t_z"] + q.CTRL


def expanding_z(p: pd.DataFrame) -> pd.DataFrame:
    """Within-country z-scores with the moments of months <= t (at least MIN_HIST months)."""
    d = p.copy()

    def ez(x: pd.Series) -> pd.Series:
        m = x.expanding(min_periods=MIN_HIST).mean()
        s = x.expanding(min_periods=MIN_HIST).std()
        return (x - m) / (s + 1e-12)

    for c in REGRESSORS:
        d[c + "_z"] = d.groupby("country")[c].transform(ez)
    g = d.groupby("country")["log_rv"]
    m = g.transform(lambda x: x.expanding(min_periods=MIN_HIST).mean())
    s = g.transform(lambda x: x.expanding(min_periods=MIN_HIST).std())
    d["log_rv_t_z"] = (d["log_rv"] - m) / (s + 1e-12)
    d["log_rv_lead1_z"] = (d["log_rv_lead1"] - m) / (s + 1e-12)
    return d


def main() -> None:
    res = q.load_json(OUT)
    p = q.load_panel()
    rt = pd.read_parquet(q.RES / "qf_a2_rt_probs.parquet")["p_rt"]
    s_rt = (rt > 0.5).astype(float).to_dict()
    pe = expanding_z(p)
    keep = pe[USED].notna().all(axis=1)          # rows with a full expanding-window history
    pf = p.copy()
    pf.loc[~keep.values, q.OUTCOME] = np.nan        # same rows, full-sample moments
    for d in (pe, pf):
        d["S_rt"] = d.index.map(s_rt)
    designs = {"expost_fullz_restricted": (pf, q.STRESS), "expost_expz": (pe, q.STRESS),
               "realtime_fullz_restricted": (pf, "S_rt"), "realtime_expz": (pe, "S_rt")}
    res["_meta"] = {"min_hist": MIN_HIST, "rows_with_history": int(keep.sum()),
                    "first_month_by_country": {c: str(pe[keep.values & (pe["country"] == c).values].index.min())[:7]
                                               for c in q.CC5},
                    "status": "exploratory (added after the analysis plan, 27/10/2026)"}
    for name, (d, scol) in designs.items():
        if name in res:
            print(f"{name}: cached")
            continue
        print(f"{name}:", flush=True)
        res[name] = a6.evaluate(d, scol)
        q.save_json(res, OUT)
    q.save_json(res, OUT)
    print("\nsummary (OLS b3, p_wild, p_MBB | DML-S delta, p_wild | DML-TA delta, p_wild | n OLS)")
    for name in designs:
        r = res[name]
        o, s, b = r["ols"], r["dml_shuffled"], r["dml_blocked"]
        print(f"  {name:27s} {o['b3']:+.3f} {o['wb_month']['p']:.3f} {o['mbb']['p']:.3f} | "
              f"{s['delta']:+.3f} {s['wild_month']['p']:.3f} | {b['delta']:+.3f} {b['wild_month']['p']:.3f} | {o['n']}")


if __name__ == "__main__":
    main()
