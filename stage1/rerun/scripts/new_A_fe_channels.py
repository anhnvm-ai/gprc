"""Phase A: FE Regression + Standardized Channel Comparison + Incremental R²

Research question: Which channel is the dominant transmission mechanism?
  Channel 1 — Volatility persistence : lagged log_rv
  Channel 2 — Geopolitical risk      : gpr_country (GPRC)
  Channel 3 — Global risk aversion   : VIX
  Channel 4 — FX shock               : dlog_fx_m

Method:
  1. Standardize all variables (z-score within country)
  2. Panel FE OLS (country FE + year FE) with clustered SE
  3. Compare standardized β → which channel has largest effect
  4. Leave-one-channel-out → Incremental R² per channel

Output:
  results/tables/tableA1_fe_standardized.csv
  results/tables/tableA2_incremental_r2.csv
  results/figures/figA_channel_ranking.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from src.utils.io import ROOT

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_monthly.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CC5 = ["VNM", "THA", "IDN", "MYS", "PHL"]
COUNTRY_LABEL = {"VNM":"Vietnam","THA":"Thailand","IDN":"Indonesia",
                 "MYS":"Malaysia","PHL":"Philippines"}

# ── Load & build panel ────────────────────────────────────────────────────────
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel = panel[panel["country"].isin(CC5)].copy()

panel["log_rv"]      = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_lag1"]  = panel["log_rv"]  # persistence = contemporaneous logRV_t (predicts logRV_{t+1})
panel["year"]         = panel.index.year

CHANNELS = {
    "log_rv_lag1":  "Volatility persistence (logRV_t)",
    "gpr_country":  "GPR country-specific",
    "vix_m":        "VIX (global risk)",
    "dlog_fx_m":    "FX shock (dlogFX)",
}
CONTROLS = ["sp500_ret_m", "us10y_m", "wti_ret_m", "ret_m"]
OUTCOME  = "log_rv_lead1"

# ── Standardize within country ────────────────────────────────────────────────
all_vars = list(CHANNELS.keys()) + CONTROLS + [OUTCOME]
for col in all_vars:
    if col in panel.columns:
        panel[col + "_z"] = panel.groupby("country")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )

# ── Build model dataset ───────────────────────────────────────────────────────
ch_z   = [c + "_z" for c in CHANNELS]
ctrl_z = [c + "_z" for c in CONTROLS if c in panel.columns]
y_col  = OUTCOME + "_z"

panel_m = panel.dropna(subset=[y_col] + ch_z).copy()

# Country dummies (FE)
country_dummies = pd.get_dummies(panel_m["country"], prefix="cty", drop_first=True)
# Year dummies (time FE)
year_dummies = pd.get_dummies(panel_m["year"], prefix="yr", drop_first=True)

Y = panel_m[y_col].values

def fit_ols(X_df, Y, label="", cov="cluster", groups=None):
    Xc = sm.add_constant(X_df.values)
    if cov == "cluster" and groups is not None:
        m = OLS(Y, Xc).fit(cov_type="cluster",
                           cov_kwds={"groups": groups})
    else:
        m = OLS(Y, Xc).fit(cov_type="HC3")
    return m

# ══════════════════════════════════════════════════════════════════════════════
# 1. FULL MODEL — all 4 channels + FE
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("MODEL FULL: 4 kenh + Country FE + Year FE + controls")
print("="*65)

X_full = pd.concat([
    panel_m[ch_z + ctrl_z].reset_index(drop=True),
    country_dummies.astype(float).reset_index(drop=True),
    year_dummies.astype(float).reset_index(drop=True),
], axis=1)

groups = panel_m["country"].values
m_full = fit_ols(X_full, Y, cov="cluster", groups=groups)

n_ch = len(ch_z)
coef_names = ["const"] + ch_z + ctrl_z + list(country_dummies.columns) + list(year_dummies.columns)

print(f"\nR² = {m_full.rsquared:.4f}   Adj-R² = {m_full.rsquared_adj:.4f}   N = {len(Y)}")
print("\n--- 4 kenh truyen dan (standardized) ---")
rows_main = []
for i, (col, label) in enumerate(CHANNELS.items()):
    idx = i + 1   # +1 for const
    b  = m_full.params[idx]
    se = m_full.bse[idx]
    t  = m_full.tvalues[idx]
    p  = m_full.pvalues[idx]
    sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
    print(f"  {label:35s}: beta={b:+.4f}  SE={se:.4f}  t={t:+.3f}  p={p:.3f} {sig}")
    rows_main.append({"channel": col, "channel_label": label,
                      "beta_std": round(b,4), "se": round(se,4),
                      "t_stat": round(t,3), "p_value": round(p,4),
                      "signif": sig, "abs_beta": abs(b)})

df_main = pd.DataFrame(rows_main).sort_values("abs_beta", ascending=False)
df_main["rank"] = range(1, len(df_main)+1)
print(f"\n--- RANKING (by |standardized beta|) ---")
for _, r in df_main.iterrows():
    print(f"  Rank {r['rank']}: {r['channel_label']:35s} |beta|={r['abs_beta']:.4f} {r['signif']}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PER-COUNTRY regressions
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PER-COUNTRY: standardized beta cho tung nuoc")
print("="*65)
rows_cc = []
for cc in CC5:
    sub = panel_m[panel_m["country"] == cc].copy()
    if len(sub) < 30:
        continue
    ys  = sub[y_col].values
    yr_d = pd.get_dummies(sub["year"], prefix="yr", drop_first=True).astype(float)
    Xcc = pd.concat([sub[ch_z + ctrl_z].reset_index(drop=True),
                     yr_d.reset_index(drop=True)], axis=1)
    try:
        mc = fit_ols(Xcc, ys, cov="HC3")
        print(f"\n  {COUNTRY_LABEL[cc]} (n={len(sub)})  R²={mc.rsquared:.3f}")
        for i, (col, label) in enumerate(CHANNELS.items()):
            b  = mc.params[i+1]
            se = mc.bse[i+1]
            p  = mc.pvalues[i+1]
            sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
            print(f"    {label:35s}: beta={b:+.4f}  p={p:.3f} {sig}")
            rows_cc.append({"country": cc, "country_label": COUNTRY_LABEL[cc],
                            "channel": col, "channel_label": label,
                            "beta_std": round(b,4), "se": round(se,4),
                            "p_value": round(p,4), "signif": sig,
                            "n_obs": len(sub)})
    except Exception as e:
        print(f"  {cc}: ERROR {e}")

df_cc = pd.DataFrame(rows_cc)


# ══════════════════════════════════════════════════════════════════════════════
# 3. INCREMENTAL R² (leave-one-channel-out)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("INCREMENTAL R²: khi bo tung kenh, R² giam bao nhieu?")
print("="*65)

fe_cols = list(country_dummies.columns) + list(year_dummies.columns)

def get_r2(channel_subset):
    Xmat = pd.concat([
        panel_m[channel_subset + ctrl_z].reset_index(drop=True),
        country_dummies.astype(float).reset_index(drop=True),
        year_dummies.astype(float).reset_index(drop=True),
    ], axis=1).astype(float)
    Xc = sm.add_constant(Xmat.values.astype(float))
    return OLS(Y, Xc).fit().rsquared

r2_full = get_r2(ch_z)
rows_r2 = []
print(f"\n  R² FULL (4 kenh): {r2_full:.4f}")
for col, label in CHANNELS.items():
    ch_minus = [c for c in ch_z if c != col + "_z"]
    r2_drop = get_r2(ch_minus)
    delta = r2_full - r2_drop
    print(f"  Bo kenh [{label:35s}]: R²={r2_drop:.4f}  ΔR²={delta:+.4f}  ({delta/r2_full*100:+.1f}%)")
    rows_r2.append({"channel": col, "channel_label": label,
                    "r2_without": round(r2_drop, 4),
                    "delta_r2": round(delta, 4),
                    "pct_contribution": round(delta/r2_full*100, 1)})

df_r2 = pd.DataFrame(rows_r2).sort_values("delta_r2", ascending=False)
df_r2["rank"] = range(1, len(df_r2)+1)


# ── Save tables ───────────────────────────────────────────────────────────────
df_main.to_csv(TABLES_DIR / "tableA1_fe_standardized.csv", index=False)
df_cc.to_csv(TABLES_DIR / "tableA1b_fe_by_country.csv", index=False)
df_r2.to_csv(TABLES_DIR / "tableA2_incremental_r2.csv", index=False)
print(f"\nSaved: tableA1_fe_standardized.csv")
print(f"Saved: tableA1b_fe_by_country.csv")
print(f"Saved: tableA2_incremental_r2.csv")


# ── FIGURE: Channel Ranking ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: standardized beta pooled
ax1 = axes[0]
colors = ["#2196F3" if p < 0.05 else "#90CAF9" for p in df_main["p_value"]]
bars = ax1.barh(df_main["channel_label"][::-1],
                df_main["abs_beta"][::-1], color=colors[::-1], alpha=0.85)
for bar, sig in zip(bars, df_main["signif"][::-1]):
    if sig:
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 sig, va="center", fontsize=9, color="#d62728")
ax1.set_xlabel("|Standardized β| (pooled FE)", fontsize=9)
ax1.set_title("Channel Ranking: Standardized Effect\n(Blue = p<0.05)", fontsize=9, fontweight="bold")
ax1.axvline(0, color="k", lw=0.5)
ax1.grid(axis="x", alpha=0.3)

# Right: incremental R²
ax2 = axes[1]
colors2 = ["#4CAF50","#FF9800","#F44336","#9C27B0"]
ax2.barh(df_r2["channel_label"][::-1],
         df_r2["pct_contribution"][::-1],
         color=colors2[:len(df_r2)][::-1], alpha=0.85)
for i, (_, row) in enumerate(df_r2[::-1].iterrows()):
    ax2.text(row["pct_contribution"] + 0.1, i,
             f"{row['pct_contribution']:+.1f}%", va="center", fontsize=8)
ax2.set_xlabel("% contribution to total R²", fontsize=9)
ax2.set_title("Channel Ranking: Incremental R²\n(R² drop when channel removed)", fontsize=9, fontweight="bold")
ax2.grid(axis="x", alpha=0.3)

fig.suptitle("Transmission Channel Ranking — 5-ASEAN Panel\n"
             "Outcome: logRV_{t+1} | Country FE + Year FE | Standardized variables",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS_DIR / "figA_channel_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figA_channel_ranking.png")

print("\nPhase A HOAN THANH.")
