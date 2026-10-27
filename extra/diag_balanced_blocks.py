"""Read-only diagnostic for the revision (26/10/2026, exploratory count cited in Sect. 6.2):
stress months per time-aware calendar block, full panel vs balanced five-market sample.
Usage: python extra/diag_balanced_blocks.py <path to stage2>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

QF = Path(sys.argv[1])
sys.path.insert(0, str(QF / "analysis"))
import qflib as q  # noqa: E402

p = q.load_panel()
# baseline sample (unbalanced): the frame used by byregime after dropna
def blocks_report(d, scol, label):
    cols = [q.OUTCOME, "gpr_country_z", scol, "log_rv_t_z"] + q.CTRL
    dd = d.dropna(subset=cols)
    months = np.array(sorted(pd.unique(dd.index)))
    blocks = np.array_split(np.arange(len(months)), 5)
    s_by_month = dd.groupby(dd.index)[scol].first()
    print(f"--- {label}: {len(dd)} obs, {len(months)} months, {int(s_by_month.sum())} stress months")
    for k, b in enumerate(blocks, start=1):
        m = months[b]
        nstress = int(s_by_month.loc[m].sum())
        nobs = int(dd.index.isin(m).sum())
        nstress_obs = int(dd[dd.index.isin(m)][scol].sum())
        print(f"  block {k}: {pd.Timestamp(m[0]).strftime('%Y-%m')} to {pd.Timestamp(m[-1]).strftime('%Y-%m')}, "
              f"{len(m)} months, {nobs} obs, stress months {nstress}, stress obs {nstress_obs}")

blocks_report(p, q.STRESS, "baseline regime, unbalanced panel")
lr = p.pivot_table(index=p.index, columns="country", values="log_rv")
months = lr.dropna().index
fb = q.fit_ms(q.regional_mean(p, months=months), starts=10)
pb = p[p.index.isin(months)].copy()
pb["S_bal_top"] = pb.index.map(q.flag_top_share(fb["smoothed"]).to_dict())
blocks_report(pb, "S_bal_top", "balanced sample, baseline stress share")
