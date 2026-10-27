"""A2 Real-time regimes (prespec P2).

For every month tau = 2013:04 ... 2026:04 the two-state Markov-switching model is re-estimated
on the regional average log RV observed up to tau (10 starts), and only the FILTERED stress
probability of month tau, P(s_tau = stress | F_tau), is kept. Stress^RT = 1{P > 0.5}
(sensitivity: 0.4, 0.6). Months before 2013:04 (the first time-aware block, used only to train
nuisance functions) take the filtered probabilities of the first estimation window.

Outputs: results/qf_a2_rt_probs.parquet (cached), results/qf_a2.json, results/figures/a2_realtime.pdf
Usage: python stage2/analysis/a2_realtime.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402

START = pd.Timestamp("2013-04-01")
CACHE = q.RES / "qf_a2_rt_probs.parquet"


def realtime_probs(p: pd.DataFrame) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    reg = q.regional_mean(p)
    rows, t0 = [], time.time()
    for tau in reg.index[reg.index >= START]:
        fm = q.fit_ms(reg[reg.index <= tau], starts=10)
        if not fm["ok"]:
            rows.append({"date": tau, "ok": False})
            continue
        if tau == START:                       # burn-in months: filtered P of the first window
            for d, v in fm["filtered"].items():
                if d < START:
                    rows.append({"date": d, "p_rt": float(v), "ok": True, "burn_in": True})
        rows.append({"date": tau, "p_rt": float(fm["filtered"].iloc[-1]), "ok": True, "burn_in": False,
                     "mu_calm": fm["mu_calm"], "mu_stress": fm["mu_stress"],
                     "p_stay_calm": fm["p_stay_calm"], "p_stay_stress": fm["p_stay_stress"],
                     "n_months": int((reg.index <= tau).sum())})
    df = pd.DataFrame(rows).set_index("date").sort_index()
    tmp = CACHE.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(CACHE)
    print(f"real-time MS: {len(df)} months in {time.time() - t0:.0f}s, failed fits: {int((~df['ok']).sum())}")
    return df


def episodes(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of 1s as (first month, last month)."""
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


def classification(rt: pd.Series, ex: pd.Series) -> dict:
    """Agreement of the real-time flag with the ex-post flag (months >= 2013:04)."""
    rt, ex = rt.astype(int), ex.reindex(rt.index).astype(int)
    tp, tn = int(((rt == 1) & (ex == 1)).sum()), int(((rt == 0) & (ex == 0)).sum())
    fp, fn = int(((rt == 1) & (ex == 0)).sum()), int(((rt == 0) & (ex == 1)).sum())
    n = len(rt)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    delays = []
    for a, b in episodes(ex):
        hit = rt[(rt.index >= a) & (rt.index <= b)]
        first = hit[hit == 1].index.min() if (hit == 1).any() else None
        delays.append({"episode_start": str(a.date()), "episode_end": str(b.date()),
                       "length": int(((ex.index >= a) & (ex.index <= b)).sum()),
                       "delay_months": None if first is None else
                       int(((rt.index >= a) & (rt.index < first)).sum())})
    false_eps = [e for e in episodes(rt) if not ex[(ex.index >= e[0]) & (ex.index <= e[1])].any()]
    return {"months": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": po,
            "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
            "episodes_ex_post": delays,
            "episodes_detected": int(sum(d["delay_months"] is not None for d in delays)),
            "false_alarm_episodes": [(str(a.date()), str(b.date())) for a, b in false_eps]}


def estimates(p: pd.DataFrame, col: str, subsample: bool) -> dict:
    d = p[p.index >= START] if subsample else p
    out = {"ols": q.ols_inference(q.design(d, col), full=True)}
    if not subsample:
        out["dml_blocked"] = q.dml_interaction(p, col, scheme="blocked")
        out["dml_shuffled"] = q.dml_interaction(p, col, scheme="shuffled")
    return out


def figure(rt: pd.DataFrame, smooth: pd.Series, ex: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for a, b in episodes(ex):
        ax.axvspan(a, b + pd.offsets.MonthEnd(0), color="0.88", lw=0)
    ax.plot(smooth.index, smooth.values, color="0.55", lw=1.2, label="ex-post smoothed probability")
    live = rt[~rt["burn_in"].astype(bool)]
    ax.plot(live.index, live["p_rt"], color="k", lw=1.2, label="real-time filtered probability")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.axvline(START, color="k", lw=0.6, ls="--")
    ax.set_ylabel("P(stress)", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fdir = q.RES / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / "a2_realtime.pdf")
    fig.savefig(fdir / "a2_realtime.png", dpi=150)


def main() -> None:
    p = q.load_panel()
    rt = realtime_probs(p)
    fm = q.fit_ms(q.regional_mean(p), starts=10)
    ex = q.flag_threshold(fm["smoothed"])
    res = {"start": str(START.date()), "n_fits": int((~rt["burn_in"].astype(bool)).sum()),
           "failed_fits": int((~rt["ok"].astype(bool)).sum())}
    live = rt[~rt["burn_in"].astype(bool)]
    for thr in (0.4, 0.5, 0.6):
        tag = f"thr{int(thr * 10):02d}"
        flag = (rt["p_rt"] > thr).astype(float)
        p[f"S_rt_{tag}"] = p.index.map(flag.to_dict())
        res[tag] = {"stress_months_live": int((live["p_rt"] > thr).sum()),
                    "classification": classification((live["p_rt"] > thr).astype(float), ex),
                    "full_sample": estimates(p, f"S_rt_{tag}", subsample=False) if thr == 0.5 else
                    {"ols": q.ols_inference(q.design(p, f"S_rt_{tag}"), full=False),
                     "dml_blocked": q.dml_interaction(p, f"S_rt_{tag}", scheme="blocked")},
                    "from_2013_04": estimates(p, f"S_rt_{tag}", subsample=True) if thr == 0.5 else None}
    # the ex-post baseline on the same samples (apples-to-apples comparison)
    res["ex_post_baseline"] = {"full_sample": estimates(p, q.STRESS, subsample=False),
                               "from_2013_04": estimates(p, q.STRESS, subsample=True)}
    res["parameter_paths"] = {k: {"min": float(live[k].min()), "max": float(live[k].max())}
                              for k in ["mu_calm", "mu_stress", "p_stay_calm", "p_stay_stress"]}
    q.save_json(res, q.RES / "qf_a2.json")
    figure(rt, fm["smoothed"], ex)
    r = res["thr05"]
    print("real-time P>0.5: live stress months", r["stress_months_live"],
          "| kappa", round(r["classification"]["cohen_kappa"], 3),
          "| episodes detected", r["classification"]["episodes_detected"], "of",
          len(r["classification"]["episodes_ex_post"]))
    for lab, blk in [("RT full", r["full_sample"]), ("RT 2013:04+", r["from_2013_04"]),
                     ("ex-post full", res["ex_post_baseline"]["full_sample"]),
                     ("ex-post 2013:04+", res["ex_post_baseline"]["from_2013_04"])]:
        o = blk["ols"]
        line = f"{lab:18s} OLS b3 {o['b3']:+.3f} DK4 p {o['dk4']['p']:.3f} fixed-b p {o['dk4_fixedb']['p']:.3f} N {o['n']}"
        if "dml_blocked" in blk:
            line += f" | TA-DML {blk['dml_blocked']['theta']:+.3f} p {blk['dml_blocked']['dk4']['p']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
