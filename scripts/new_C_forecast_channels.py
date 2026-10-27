"""Phase C: Forecasting — which channel improves OOS prediction of logRV?

Model specs (expanding window, OOS from 2020-01):
  M0: Baseline — logRV_lag1 only (HAR-minimal)
  M1: M0 + GPR_country
  M2: M0 + VIX
  M3: M0 + dlogFX
  M4: M0 + GPR + VIX + FX  (full transmission)
  M5: M0 + all macro controls (Ridge)

Metrics: RMSE, R²_OOS (Campbell-Thompson), Clark-West vs M0

Output:
  results/tables/tableC1_forecast_channels.csv
  results/tables/tableC2_cw_tests.csv
  results/figures/figC_forecast_r2oos.png
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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils.io import ROOT

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_monthly.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CC5 = ["VNM","THA","IDN","MYS","PHL"]
COUNTRY_LABEL = {"VNM":"Vietnam","THA":"Thailand","IDN":"Indonesia",
                 "MYS":"Malaysia","PHL":"Philippines"}
OOS_START = "2020-01-01"
MIN_TRAIN = 60

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel = panel[panel["country"].isin(CC5)].copy()

panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_lag1"]  = panel["log_rv"]  # persistence = contemporaneous logRV_t (predicts logRV_{t+1})

OUTCOME = "log_rv_lead1"
MODEL_SPECS = {
    "M0_Persistence":   ["log_rv_lag1"],
    "M1_Pers+GPR":      ["log_rv_lag1","gpr_country"],
    "M2_Pers+VIX":      ["log_rv_lag1","vix_m"],
    "M3_Pers+FX":       ["log_rv_lag1","dlog_fx_m"],
    "M4_Full_3ch":      ["log_rv_lag1","gpr_country","vix_m","dlog_fx_m"],
    "M5_Full+Macro":    ["log_rv_lag1","gpr_country","vix_m","dlog_fx_m",
                         "sp500_ret_m","us10y_m","wti_ret_m"],
}

def rmse(y, yhat):   return np.sqrt(np.nanmean((y-yhat)**2))
def r2_oos(y, yhat, bm):
    ss_r = np.nansum((y-yhat)**2)
    ss_b = np.nansum((y-bm)**2)
    return 1 - ss_r/ss_b if ss_b > 0 else np.nan

def clark_west(y, yhat1, yhat2):
    e1 = np.asarray(y) - np.asarray(yhat1)
    e2 = np.asarray(y) - np.asarray(yhat2)
    f  = e1**2 - (e2**2 - (yhat1-yhat2)**2)
    f  = f[~np.isnan(f)]
    mu = f.mean(); se = np.std(f,ddof=1)/np.sqrt(len(f))
    t  = mu/se if se>0 else 0
    return float(t), float(1-norm.cdf(t))

def rolling_oos_logRV(df_full, feat_cols, outcome, model="ols"):
    result = pd.Series(np.nan, index=df_full.index)
    df_cc  = df_full.dropna(subset=[outcome]+feat_cols).copy()
    oos_mask = df_cc.index >= OOS_START

    for t_idx in np.where(oos_mask)[0]:
        train = df_cc.iloc[:t_idx]
        if len(train) < MIN_TRAIN:
            continue
        Y_tr = train[outcome].values
        X_tr = train[feat_cols].values
        X_te = df_cc.iloc[[t_idx]][feat_cols].values

        if model == "ols":
            from numpy.linalg import lstsq
            Xc = np.column_stack([np.ones(len(X_tr)), X_tr])
            beta, *_ = lstsq(Xc, Y_tr, rcond=None)
            pred = float(np.column_stack([np.ones(1), X_te]) @ beta)
        elif model == "ridge":
            pipe = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(cv=5))])
            pipe.fit(X_tr, Y_tr)
            pred = float(pipe.predict(X_te))
        else:
            pred = float(np.mean(Y_tr))

        result.iloc[df_full.index.get_loc(df_cc.index[t_idx])] = pred

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
all_metric = []
all_cw     = []

for cc in CC5:
    sub = panel[panel["country"] == cc].copy()
    print(f"\n{'='*55}")
    print(f"{COUNTRY_LABEL[cc]}")
    print("="*55)

    preds_dict = {}
    for mname, feats in MODEL_SPECS.items():
        feat_ok = [f for f in feats if f in sub.columns and sub[f].notna().sum() > 30]
        if not feat_ok:
            continue
        alg = "ridge" if "Macro" in mname else "ols"
        preds = rolling_oos_logRV(sub, feat_ok, OUTCOME, model=alg)
        preds_dict[mname] = preds

        oos_mask = (sub.index >= OOS_START) & sub[OUTCOME].notna()
        y_oos    = sub.loc[oos_mask, OUTCOME].values
        yh_oos   = preds[oos_mask].values
        valid    = ~np.isnan(yh_oos) & ~np.isnan(y_oos)
        y_v = y_oos[valid]; yh_v = yh_oos[valid]
        if valid.sum() < 10:
            continue

        bm_mean = np.full(valid.sum(), sub.loc[sub.index < OOS_START, OUTCOME].mean())
        rmse_v  = rmse(y_v, yh_v)
        r2_v    = r2_oos(y_v, yh_v, bm_mean)
        print(f"  {mname:20s}: RMSE={rmse_v:.4f}  R²_OOS={r2_v:+.3f}  n={valid.sum()}")
        all_metric.append({"country":cc,"country_label":COUNTRY_LABEL[cc],
                           "model":mname,"RMSE":round(rmse_v,4),
                           "R2_OOS":round(r2_v,4),"n_oos":int(valid.sum())})

    # Clark-West vs M0
    bm = "M0_Persistence"
    if bm in preds_dict:
        oos_mask = (sub.index >= OOS_START) & sub[OUTCOME].notna()
        y_oos    = sub.loc[oos_mask, OUTCOME].values
        yh_bm    = preds_dict[bm][oos_mask].values

        for mname, preds in preds_dict.items():
            if mname == bm: continue
            yh_alt = preds[oos_mask].values
            valid  = ~np.isnan(yh_bm) & ~np.isnan(yh_alt) & ~np.isnan(y_oos)
            if valid.sum() < 10: continue
            cw_t, cw_p = clark_west(y_oos[valid], yh_bm[valid], yh_alt[valid])
            sig = "***" if cw_p<0.01 else "**" if cw_p<0.05 else "*" if cw_p<0.10 else ""
            print(f"  CW [{mname}]: t={cw_t:+.3f}  p={cw_p:.3f} {sig}")
            all_cw.append({"country":cc,"benchmark":bm,"challenger":mname,
                           "CW_t":round(cw_t,3),"CW_p":round(cw_p,3),"CW_sig":sig})

df_metric = pd.DataFrame(all_metric)
df_cw     = pd.DataFrame(all_cw)

df_metric.to_csv(TABLES_DIR / "tableC1_forecast_channels.csv", index=False)
df_cw.to_csv(TABLES_DIR / "tableC2_cw_tests.csv", index=False)
print(f"\nSaved: tableC1_forecast_channels.csv")
print(f"Saved: tableC2_cw_tests.csv")


# ── FIGURE: R²_OOS by model and country ──────────────────────────────────────
if not df_metric.empty:
    model_list = list(MODEL_SPECS.keys())
    colors = {"VNM":"#e41a1c","THA":"#377eb8","IDN":"#4daf4a",
               "MYS":"#984ea3","PHL":"#ff7f00"}
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(model_list))
    width = 0.16

    for k, cc in enumerate(CC5):
        sub_m = df_metric[df_metric["country"]==cc]
        r2vals = [sub_m.loc[sub_m["model"]==m,"R2_OOS"].values[0]
                  if len(sub_m[sub_m["model"]==m])>0 else np.nan
                  for m in model_list]
        off = (k - len(CC5)/2 + 0.5)*width
        ax.bar(x+off, r2vals, width, color=colors[cc], alpha=0.8,
               label=COUNTRY_LABEL[cc])

    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_"," ") for m in model_list],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("R²_OOS (Campbell-Thompson 2008)", fontsize=9)
    ax.set_title("Forecast Gain: Which Channel Improves OOS Prediction?\n"
                 "Baseline = Volatility Persistence only  |  OOS from 2020-01",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "figC_forecast_r2oos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figC_forecast_r2oos.png")

print("\nPhase C HOAN THANH.")
