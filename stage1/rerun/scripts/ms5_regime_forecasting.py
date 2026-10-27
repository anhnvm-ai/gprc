"""MS5: Regime-Conditional Out-of-Sample Forecasting (Step 5 of upgrade guide)

Expanding-window forecast of logRV_{t+1}; evaluate accuracy SEPARATELY in
calm vs stress OOS months. Tests whether GPRC / VIX / FX improve forecasts
specifically during stress.

Models:
  M0: Persistence only (logRV_t)
  M1: + GPRC
  M2: + VIX
  M3: + FX
  M4: Full (GPRC+VIX+FX)

Metrics per regime: RMSE, R2_OOS (vs persistence benchmark), Clark-West.

Outputs:
  results/tables/tableMS5_regime_forecast.csv
  results/tables/tableMS5_cw_by_regime.csv
  results/figures/figMS5_regime_forecast.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.utils.io import ROOT

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_regimes.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CC5 = ["VNM","THA","IDN","MYS","PHL"]
COUNTRY_LABEL = {"VNM":"Vietnam","THA":"Thailand","IDN":"Indonesia",
                 "MYS":"Malaysia","PHL":"Philippines"}
OOS_START = "2020-01-01"
MIN_TRAIN = 60
STRESS = "stress_common"

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_t"]     = panel["log_rv"]

OUTCOME = "log_rv_lead1"
SPECS = {
    "M0_Persistence": ["log_rv_t"],
    "M1_+GPRC":       ["log_rv_t","gpr_country"],
    "M2_+VIX":        ["log_rv_t","vix_m"],
    "M3_+FX":         ["log_rv_t","dlog_fx_m"],
    "M4_Full":        ["log_rv_t","gpr_country","vix_m","dlog_fx_m"],
}

def rmse(y, yh): return np.sqrt(np.nanmean((y-yh)**2))
def r2_oos(y, yh, bm):
    sr = np.nansum((y-yh)**2); sb = np.nansum((y-bm)**2)
    return 1 - sr/sb if sb>0 else np.nan
def clark_west(y, yh1, yh2):
    e1 = y-yh1; e2 = y-yh2
    f  = e1**2 - (e2**2 - (yh1-yh2)**2)
    f  = f[~np.isnan(f)]
    if len(f) < 5: return np.nan, np.nan
    mu = f.mean(); se = np.std(f,ddof=1)/np.sqrt(len(f))
    t = mu/se if se>0 else 0
    return float(t), float(1-norm.cdf(t))

def rolling_oos(df_full, feats, outcome):
    res = pd.Series(np.nan, index=df_full.index)
    d = df_full.dropna(subset=[outcome]+feats).copy()
    mask = d.index >= OOS_START
    from numpy.linalg import lstsq
    for ti in np.where(mask)[0]:
        tr = d.iloc[:ti]
        if len(tr) < MIN_TRAIN: continue
        Xc = np.column_stack([np.ones(len(tr)), tr[feats].values])
        beta, *_ = lstsq(Xc, tr[outcome].values, rcond=None)
        xt = np.column_stack([np.ones(1), d.iloc[[ti]][feats].values])
        res.iloc[df_full.index.get_loc(d.index[ti])] = float(xt @ beta)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Collect predictions across all countries, then split by regime
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("MS5: REGIME-CONDITIONAL OOS FORECASTING")
print("="*65)

# accumulate pooled prediction errors by regime
pooled = {m: {"y":[], "yh":[], "stress":[]} for m in SPECS}

for cc in CC5:
    sub = panel[panel["country"]==cc].copy()
    for m, feats in SPECS.items():
        fok = [f for f in feats if f in sub.columns and sub[f].notna().sum()>30]
        if not fok: continue
        preds = rolling_oos(sub, fok, OUTCOME)
        mask = (sub.index>=OOS_START) & sub[OUTCOME].notna()
        y  = sub.loc[mask, OUTCOME].values
        yh = preds[mask].values
        st = sub.loc[mask, STRESS].values
        ok = ~np.isnan(yh) & ~np.isnan(y) & ~np.isnan(st)
        pooled[m]["y"].extend(y[ok]); pooled[m]["yh"].extend(yh[ok])
        pooled[m]["stress"].extend(st[ok])

for m in SPECS:
    for k in pooled[m]:
        pooled[m][k] = np.array(pooled[m][k])

# Benchmark = M0 persistence
bm_y  = pooled["M0_Persistence"]["y"]
bm_yh = pooled["M0_Persistence"]["yh"]
bm_st = pooled["M0_Persistence"]["stress"]

rows_metric, rows_cw = [], []
for regime_val, regime_label in [(None,"All"), (0,"Calm"), (1,"Stress")]:
    print(f"\n  --- {regime_label} OOS months ---")
    for m in SPECS:
        y, yh, st = pooled[m]["y"], pooled[m]["yh"], pooled[m]["stress"]
        if regime_val is None:
            sel = np.ones(len(y), bool)
        else:
            sel = (st == regime_val)
        if sel.sum() < 8:
            continue
        # unconditional benchmark mean for R2_OOS uses persistence model M0 errors?
        # Campbell-Thompson: benchmark = historical mean. We use M0 as challenger ref for CW.
        y_s, yh_s = y[sel], yh[sel]
        bm_mean = np.full(sel.sum(), np.mean(y_s))
        r2 = r2_oos(y_s, yh_s, bm_mean)
        rm = rmse(y_s, yh_s)
        print(f"    {m:16s}: RMSE={rm:.4f}  R2_OOS(vs mean)={r2:+.3f}  n={sel.sum()}")
        rows_metric.append({"regime":regime_label,"model":m,"RMSE":round(rm,4),
                            "R2_OOS":round(r2,4),"n":int(sel.sum())})
        # Clark-West vs M0 (same regime selection)
        if m != "M0_Persistence":
            sel0 = (bm_st==regime_val) if regime_val is not None else np.ones(len(bm_y),bool)
            cwt, cwp = clark_west(y[sel], bm_yh[sel0], yh[sel])
            sig = "***" if cwp<0.01 else "**" if cwp<0.05 else "*" if cwp<0.10 else ""
            print(f"        CW vs M0: t={cwt:+.3f} p={cwp:.3f} {sig}")
            rows_cw.append({"regime":regime_label,"challenger":m,
                            "CW_t":round(cwt,3),"CW_p":round(cwp,3),"CW_sig":sig})

df_metric = pd.DataFrame(rows_metric)
df_cw     = pd.DataFrame(rows_cw)
df_metric.to_csv(TABLES_DIR / "tableMS5_regime_forecast.csv", index=False)
df_cw.to_csv(TABLES_DIR / "tableMS5_cw_by_regime.csv", index=False)
print(f"\nSaved: tableMS5_regime_forecast.csv, tableMS5_cw_by_regime.csv")


# ── FIGURE ────────────────────────────────────────────────────────────────────
if not df_metric.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    model_list = list(SPECS.keys())

    # Left: RMSE by regime
    ax0 = axes[0]
    x = np.arange(len(model_list)); w = 0.25
    for i,(reg,color) in enumerate([("All","#888"),("Calm","#4daf4a"),("Stress","#d62728")]):
        sub = df_metric[df_metric["regime"]==reg]
        vals = [sub.loc[sub["model"]==m,"RMSE"].values[0] if len(sub[sub["model"]==m]) else np.nan
                for m in model_list]
        ax0.bar(x+(i-1)*w, vals, w, color=color, alpha=0.8, label=reg)
    ax0.set_xticks(x); ax0.set_xticklabels([m.replace("_","\n") for m in model_list], fontsize=7)
    ax0.set_ylabel("RMSE (logRV)", fontsize=9)
    ax0.set_title("Forecast RMSE by Regime", fontsize=10, fontweight="bold")
    ax0.legend(fontsize=8); ax0.grid(axis="y", alpha=0.3)

    # Right: Clark-West significance heatmap-ish
    ax1 = axes[1]
    challengers = [m for m in model_list if m!="M0_Persistence"]
    regimes = ["All","Calm","Stress"]
    mat = np.full((len(challengers), len(regimes)), np.nan)
    for i,ch in enumerate(challengers):
        for j,reg in enumerate(regimes):
            r = df_cw[(df_cw["challenger"]==ch)&(df_cw["regime"]==reg)]
            if len(r): mat[i,j] = r["CW_t"].iloc[0]
    im = ax1.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=3)
    ax1.set_xticks(range(len(regimes))); ax1.set_xticklabels(regimes, fontsize=9)
    ax1.set_yticks(range(len(challengers)))
    ax1.set_yticklabels([c.replace("_"," ") for c in challengers], fontsize=8)
    for i in range(len(challengers)):
        for j in range(len(regimes)):
            if not np.isnan(mat[i,j]):
                r = df_cw[(df_cw["challenger"]==challengers[i])&(df_cw["regime"]==regimes[j])]
                sig = r["CW_sig"].iloc[0] if len(r) else ""
                ax1.text(j,i,f"{mat[i,j]:.2f}\n{sig}",ha="center",va="center",fontsize=8,
                         color="black")
    plt.colorbar(im, ax=ax1, label="Clark-West t-stat", shrink=0.8)
    ax1.set_title("Clark-West vs Persistence Benchmark\n(higher = channel improves forecast)",
                  fontsize=10, fontweight="bold")

    fig.suptitle("Regime-Conditional Forecasting — 5-ASEAN (OOS from 2020-01)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "figMS5_regime_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figMS5_regime_forecast.png")

print("\nMS5 HOAN THANH.")
