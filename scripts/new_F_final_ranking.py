"""Phase F — Final Channel Ranking Table + GPR as Regime Amplifier

Combines results from A, B, C, D into one master summary table.
Adds "Interpretation" column.
Reframes GPR as regime-dependent amplifier, not main channel.

Output:
  results/tables/tableF_master_channel_ranking.xlsx
  results/figures/figF_master_ranking.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.utils.io import ROOT

TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"

def load(f):
    p = TABLES_DIR / f
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

dfA1 = load("tableA1_fe_standardized.csv")
dfA2 = load("tableA2_incremental_r2.csv")
dfB1 = load("tableB1_dml_channels_pooled.csv")
dfB2 = load("tableB2_dml_channels_by_country.csv")
dfC1 = load("tableC1_forecast_channels.csv")
dfC2 = load("tableC2_cw_tests.csv")

CH_ORDER = ["log_rv_lag1","gpr_country","vix_m","dlog_fx_m"]
CH_LABEL = {
    "log_rv_lag1": "Volatility Persistence",
    "gpr_country": "GPR Country (regime amplifier)",
    "vix_m":       "VIX / Global Risk Aversion",
    "dlog_fx_m":   "FX Shock",
}
CH_SHORT = {
    "log_rv_lag1": "Persistence",
    "gpr_country": "GPR Country",
    "vix_m":       "VIX",
    "dlog_fx_m":   "FX",
}

print("="*75)
print("MASTER CHANNEL RANKING TABLE")
print("="*75)

rows = []
for ch in CH_ORDER:
    label = CH_LABEL[ch]
    r = {"channel": ch, "channel_label": label}

    # FE standardized beta
    if not dfA1.empty and ch in dfA1["channel"].values:
        row = dfA1[dfA1["channel"]==ch].iloc[0]
        r["fe_beta"]  = row["beta_std"]
        r["fe_p"]     = row["p_value"]
        r["fe_sig"]   = row["signif"] if pd.notna(row["signif"]) else ""
    else:
        r["fe_beta"] = r["fe_p"] = r["fe_sig"] = np.nan

    # Incremental R²
    if not dfA2.empty and ch in dfA2["channel"].values:
        row2 = dfA2[dfA2["channel"]==ch].iloc[0]
        r["delta_r2_pct"] = row2["pct_contribution"]
        r["delta_r2_rank"] = row2["rank"]
    else:
        r["delta_r2_pct"] = r["delta_r2_rank"] = np.nan

    # DML theta (Lasso, pooled)
    if not dfB1.empty:
        sub = dfB1[(dfB1["channel"]==ch) & (dfB1["ml"]=="lasso") & (dfB1["country"]=="POOLED")]
        if len(sub):
            r["dml_theta"] = sub["theta"].iloc[0]
            r["dml_p"]     = sub["p_value"].iloc[0]
            r["dml_sig"]   = sub["signif"].iloc[0] if pd.notna(sub["signif"].iloc[0]) else ""

    # Sign indicators
    r["fe_sign"]  = "+" if r.get("fe_beta", 0) > 0 else "−"
    r["dml_sign"] = "+" if r.get("dml_theta", 0) > 0 else "−"

    # Country support: how many of 5 countries have DML p<0.10 (Lasso)
    if not dfB2.empty:
        sub_cc = dfB2[(dfB2["channel"]==ch) & (dfB2["ml"]=="lasso")]
        n_sig = (sub_cc["p_value"] < 0.10).sum()
        r["country_support"] = f"{n_sig}/5"
    else:
        r["country_support"] = "—"

    # Forecast: how much does adding this channel improve R²_OOS vs M0?
    # Compare M1 (+GPR), M2 (+VIX), M3 (+FX)
    forecast_map = {
        "gpr_country": "M1_Pers+GPR",
        "vix_m":       "M2_Pers+VIX",
        "dlog_fx_m":   "M3_Pers+FX",
        "log_rv_lag1": "M0_Persistence",
    }
    if not dfC1.empty and ch in forecast_map:
        mname = forecast_map[ch]
        sub_c = dfC1[dfC1["model"]==mname]
        if len(sub_c):
            avg_r2 = sub_c["R2_OOS"].mean()
            r["avg_r2oos"] = round(avg_r2, 3)
        if ch != "log_rv_lag1" and not dfC2.empty:
            cw_sigs = dfC2[dfC2["challenger"]==mname]["CW_sig"].values
            n_sig = (cw_sigs != "").sum() if len(cw_sigs) else 0
            r["cw_countries"] = f"{n_sig}/5"

    rows.append(r)

df_rank = pd.DataFrame(rows)

# Interpretation column
interp = {
    "log_rv_lag1": "MAIN channel. Highly persistent; ARCH/HAR dynamics dominate.",
    "vix_m":       "SECONDARY channel. Global risk-off transmits into ASEAN. DML confirms positive causal effect despite FE sign flip (year FE collinearity).",
    "dlog_fx_m":   "CONDITIONAL channel. Significant in IDN, MYS (flexible FX); weak elsewhere.",
    "gpr_country": "REGIME AMPLIFIER. Baseline effect small. Amplifies volatility after COVID (VNM GPR×PostCOVID θ=+0.27★★★). Country-specific, not global.",
}
df_rank["interpretation"] = df_rank["channel"].map(interp)

# Print
print()
print(f"{'Channel':<30} {'FE sign':>7} {'FE β':>8} {'DML sign':>8} {'DML θ':>8} {'DML p':>7} "
      f"{'ΔR²%':>6} {'R²_OOS':>7} {'CW':>5} {'Ctry':>5}")
print("─"*100)
for _, r in df_rank.iterrows():
    fe_b   = r['fe_beta'] if pd.notna(r.get('fe_beta')) else float('nan')
    dml_t  = r.get('dml_theta', float('nan'))
    dml_p  = r.get('dml_p', float('nan'))
    dr2    = r.get('delta_r2_pct', float('nan'))
    r2oos  = r.get('avg_r2oos', float('nan'))
    fe_sig_str  = r.get('fe_sig','') or 'n.s.'
    dml_sig_str = r.get('dml_sig','') or 'n.s.'
    print(f"  {r['channel_label']:<28} {r.get('fe_sign','?'):>7}({fe_sig_str:<4}) "
          f"{fe_b:>+8.4f}  {r.get('dml_sign','?'):>6}({dml_sig_str:<4}) "
          f"{dml_t:>+8.4f}  {dml_p:>7.4f}  {dr2:>5.1f}%  "
          f"{r2oos:>+7.3f}  {r.get('cw_countries','—'):>5}  {r.get('country_support','—'):>4}")
    print(f"     → {r['interpretation']}")
    print()

# Save xlsx
out = TABLES_DIR / "tableF_master_channel_ranking.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as w:
    df_rank.to_excel(w, sheet_name="Channel Ranking", index=False)
    ws = w.sheets["Channel Ranking"]
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["M"].width = 70
print(f"\nSaved: {out}")


# ── FIGURE: Master 3-panel ranking ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ch_labels = [CH_SHORT[c] for c in CH_ORDER]
colors = ["#2196F3","#FF5722","#4CAF50","#9C27B0"]

# Panel 1: DML theta
ax = axes[0]
thetas = [df_rank.loc[df_rank["channel"]==c,"dml_theta"].values[0]
          if "dml_theta" in df_rank.columns and len(df_rank[df_rank["channel"]==c]) else 0
          for c in CH_ORDER]
sigs = [df_rank.loc[df_rank["channel"]==c,"dml_sig"].values[0]
        if "dml_sig" in df_rank.columns and len(df_rank[df_rank["channel"]==c]) else ""
        for c in CH_ORDER]
bars = ax.barh(ch_labels[::-1], [abs(t) for t in thetas[::-1]],
               color=colors[::-1], alpha=0.85)
for bar, sig, t in zip(bars, sigs[::-1], thetas[::-1]):
    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
            f"{t:+.3f}{sig}", va="center", fontsize=8)
ax.set_xlabel("|DML θ| pooled (std)", fontsize=9)
ax.set_title("DML Causal Effect\n(orthogonal, Lasso)", fontsize=9, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Panel 2: Incremental R²
ax = axes[1]
dr2 = [df_rank.loc[df_rank["channel"]==c,"delta_r2_pct"].values[0]
       if len(df_rank[df_rank["channel"]==c]) else 0
       for c in CH_ORDER]
ax.barh(ch_labels[::-1], dr2[::-1], color=colors[::-1], alpha=0.85)
for i, (v, label) in enumerate(zip(dr2[::-1], ch_labels[::-1])):
    ax.text(v+0.05, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("ΔR² when channel removed (%)", fontsize=9)
ax.set_title("Incremental R²\n(leave-one-out)", fontsize=9, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Panel 3: Forecast R²_OOS
ax = axes[2]
r2oos_vals = [df_rank.loc[df_rank["channel"]==c,"avg_r2oos"].values[0]
              if "avg_r2oos" in df_rank.columns and len(df_rank[df_rank["channel"]==c]) else 0
              for c in CH_ORDER]
cw_labs = [df_rank.loc[df_rank["channel"]==c,"cw_countries"].values[0]
           if "cw_countries" in df_rank.columns and len(df_rank[df_rank["channel"]==c]) else "—"
           for c in CH_ORDER]
ax.barh(ch_labels[::-1], r2oos_vals[::-1], color=colors[::-1], alpha=0.85)
for i, (v, cw) in enumerate(zip(r2oos_vals[::-1], cw_labs[::-1])):
    ax.text(max(v,0)+0.001, i, f"{v:+.3f}  CW={cw}", va="center", fontsize=8)
ax.set_xlabel("Avg R²_OOS (5-ASEAN)", fontsize=9)
ax.set_title("Forecasting Gain\n(OOS from 2020-01)", fontsize=9, fontweight="bold")
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.grid(axis="x", alpha=0.3)

# Legend
patches = [mpatches.Patch(color=colors[i], label=CH_SHORT[c], alpha=0.85)
           for i, c in enumerate(CH_ORDER)]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))

fig.suptitle(
    "Transmission Channel Ranking — 5-ASEAN (2010–2026)\n"
    "Volatility Persistence > VIX > FX > GPR (regime amplifier)\n"
    "DML partialled out other channels  |  Outcome: logRV_{t+1}",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig(FIGS_DIR / "figF_master_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figF_master_ranking.png")

print("\nPhase F HOAN THANH.")
