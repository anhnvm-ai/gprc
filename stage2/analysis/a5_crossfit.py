"""A5 Cross-fitting schemes for the DML estimate of GPRC x Stress (prespec P7), Lasso, C0 controls.

Schemes: shuffled; forward-blocked; forward-blocked with a gap of g = 1 or 3 months; neighbour
deletion (l = 3 months each side); reverse cross-fitting (Ciganovic, D'Amario & Tancioni 2026).
K in {5, 8, 10}. Every result reports theta, the month-aggregated DK(4) SE and N.

Output: results/qf_a5.json, results/tables/a5_crossfit.csv
Usage: python stage2/analysis/a5_crossfit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402

VARIANTS = [("shuffled", {}), ("blocked", {}), ("blocked_gap", {"gap": 1}), ("blocked_gap", {"gap": 3}),
            ("neighbour", {"nbr": 3}), ("reverse", {}),
            # EXPLORATORY (added after seeing reverse|K8 = +2.68): reverse training sides but a
            # pooled slope instead of the average of fold slopes, to separate the two choices
            ("reverse", {"pool": "pooled"})]


def label(sch: str, kw: dict) -> str:
    return sch + "".join(f"_{k}{v}" for k, v in kw.items())


def main() -> None:
    out_path = q.RES / "qf_a5.json"
    res = q.load_json(out_path)
    p = q.load_panel()
    for K in (5, 8, 10):
        for sch, kw in VARIANTS:
            key = f"{label(sch, kw)}|K{K}"
            if key in res:
                continue
            r = q.dml_interaction(p, q.STRESS, scheme=sch, K=K, **kw)
            res[key] = {k: r[k] for k in ["theta", "n", "iid", "dk4", "dk6", "dk12", "r2_y", "r2_d",
                                          "pool", "fold_thetas"]}
            res[key]["exploratory"] = kw.get("pool") == "pooled"
            q.save_json(res, out_path)
            print(f"{key:24s} theta {r['theta']:+.3f}  DK4 SE {r['dk4']['se']:.3f} p {r['dk4']['p']:.3f}  N {r['n']}",
                  flush=True)
    rows = [{"scheme": k.split("|")[0], "K": int(k.split("|K")[1]), "theta": round(v["theta"], 4),
             "se_dk4": round(v["dk4"]["se"], 4), "p_dk4": round(v["dk4"]["p"], 4), "n": v["n"]}
            for k, v in res.items()]
    (q.RES / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["scheme", "K"]).to_csv(q.RES / "tables" / "a5_crossfit.csv", index=False)


if __name__ == "__main__":
    main()
