"""EDA: ADF, KPSS, Granger causality, summary stats — outputs paste-ready CSVs."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from src.utils.io import ROOT, ensure_dir

def adf_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna()
        if len(s) < 50: continue
        try:
            stat, p, *_ = adfuller(s, autolag="AIC")
            rows.append((c, stat, p, "stationary" if p < 0.05 else "non-stationary"))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["series","ADF_stat","p_value","decision_5pct"])

def granger_table(df: pd.DataFrame, target: str, candidates: list[str], maxlag: int = 5) -> pd.DataFrame:
    rows = []
    for c in candidates:
        if c not in df.columns or c == target: continue
        sub = df[[target, c]].dropna()
        if len(sub) < 100: continue
        try:
            res = grangercausalitytests(sub, maxlag=maxlag, verbose=False)
            pvals = [res[L][0]["ssr_ftest"][1] for L in range(1, maxlag+1)]
            rows.append((c, target, *pvals, min(pvals)))
        except Exception as e:
            rows.append((c, target, *([np.nan]*maxlag), np.nan))
    cols = ["cause","effect"] + [f"p_lag{L}" for L in range(1, maxlag+1)] + ["p_min"]
    return pd.DataFrame(rows, columns=cols)

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe().T
    desc["skew"]  = df.skew(numeric_only=True)
    desc["kurt"]  = df.kurt(numeric_only=True)
    return desc

def save_eda(df: pd.DataFrame, target: str, candidates: list[str], outdir: str | Path | None = None) -> None:
    outdir = Path(outdir) if outdir else ROOT / "results" / "tables"
    ensure_dir(outdir)
    summary_stats(df).to_csv(outdir / "table01_summary_stats.csv")
    adf_table(df).to_csv(outdir / "table03_adf.csv", index=False)
    granger_table(df, target, candidates).to_csv(outdir / "table04_granger.csv", index=False)
    df.corr(numeric_only=True).to_csv(outdir / "table02_correlation.csv")
    print(f"[EDA] tables saved to {outdir}")
