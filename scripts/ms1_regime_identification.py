"""MS1: Markov-Switching Regime Identification (Step 1 of upgrade guide)

Identifies latent volatility regimes (Calm vs Stress) endogenously from data.

Two regime definitions:
  A) COMMON ASEAN regime  — from cross-country average log_rv (balanced, primary)
  B) COUNTRY-specific      — per-country MS (shows heterogeneity, secondary)

Model: log RV_t = mu_{s_t} + eps_t  (2-state Markov, switching mean/variance)
Output: smoothed P(stress), Stress_t flag merged back into panel.

Outputs:
  data/processed/asean_panel_regimes.parquet   (panel + stress flags)
  results/tables/tableMS1_regime_stats.csv
  results/figures/figMS1_regime_probabilities.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15,'axes.titlesize':15,'axes.labelsize':14,'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':12.5,'figure.dpi':110,'savefig.dpi':300,'savefig.bbox':'tight'})  # PUBSTYLE
from src.utils.io import ROOT
from src.models.markov_regime import fit_markov_2state

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_monthly.parquet"
OUT_PANEL  = ROOT / "data" / "processed" / "asean_panel_regimes.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CC5 = ["VNM","THA","IDN","MYS","PHL"]
COUNTRY_LABEL = {"VNM":"Vietnam","THA":"Thailand","IDN":"Indonesia",
                 "MYS":"Malaysia","PHL":"Philippines"}
CC_COLORS = {"VNM":"#e41a1c","THA":"#377eb8","IDN":"#4daf4a",
             "MYS":"#984ea3","PHL":"#ff7f00"}

# Known crisis windows for annotation
CRISES = [
    ("2011-08-01","2011-10-01","EU debt"),
    ("2013-05-01","2013-09-01","Taper"),
    ("2015-08-01","2016-02-01","CNY deval"),
    ("2018-10-01","2018-12-01","Trade war"),
    ("2020-02-01","2020-12-01","COVID"),
    ("2022-02-01","2022-06-01","RU-UA"),
]

# ── Load panel ────────────────────────────────────────────────────────────────
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel = panel[panel["country"].isin(CC5)].copy()
panel["log_rv"] = np.log(panel["rv_m"].clip(lower=1e-8))

stats_rows = []

# ══════════════════════════════════════════════════════════════════════════════
# A. COMMON ASEAN REGIME (from regional average log_rv)
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("A. COMMON ASEAN REGIME (cross-country average log_rv)")
print("="*65)
regional = panel.groupby(panel.index)["log_rv"].mean().dropna()
res_common = fit_markov_2state(regional, seed=42)

print(f"  switching_variance = {res_common.switching_variance}   LLF = {res_common.llf:.1f}")
print(f"  mean log_rv: calm = {res_common.mu_calm:.3f}  stress = {res_common.mu_stress:.3f}")
print(f"  -> RV level: calm = {np.exp(res_common.mu_calm):.3f}  stress = {np.exp(res_common.mu_stress):.3f}")
print(f"  Stress months: {res_common.n_stress}/{res_common.n_total} ({100*res_common.n_stress/res_common.n_total:.0f}%)")
print(f"  P(stay calm) = {res_common.p_stay_calm:.2f}  -> expected calm duration = {res_common.dur_calm:.1f} months")
print(f"  P(stay stress) = {res_common.p_stay_stress:.2f}  -> expected stress duration = {res_common.dur_stress:.1f} months")

stress_dates = res_common.stress_flag[res_common.stress_flag == 1].index
print(f"  Stress months by year: {dict(pd.Series(stress_dates.year).value_counts().sort_index())}")

stats_rows.append({
    "regime_def":"COMMON_ASEAN","unit":"regional_avg",
    "switching_var":res_common.switching_variance,"llf":round(res_common.llf,2),
    "mu_calm":round(res_common.mu_calm,3),"mu_stress":round(res_common.mu_stress,3),
    "rv_calm":round(np.exp(res_common.mu_calm),3),"rv_stress":round(np.exp(res_common.mu_stress),3),
    "n_stress":res_common.n_stress,"n_total":res_common.n_total,
    "pct_stress":round(100*res_common.n_stress/res_common.n_total,1),
    "p_stay_calm":round(res_common.p_stay_calm,3),"p_stay_stress":round(res_common.p_stay_stress,3),
    "dur_calm":round(res_common.dur_calm,1),"dur_stress":round(res_common.dur_stress,1),
})

# Build common stress flag mapped to all dates
common_stress = res_common.stress_flag.reindex(panel.index.unique()).fillna(0).astype(int)
common_pstress = res_common.smoothed_stress.reindex(panel.index.unique())

# ══════════════════════════════════════════════════════════════════════════════
# B. COUNTRY-SPECIFIC REGIMES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("B. COUNTRY-SPECIFIC REGIMES")
print("="*65)
country_results = {}
for cc in CC5:
    sub = panel[panel["country"] == cc].copy()
    y = sub["log_rv"].dropna()
    try:
        res = fit_markov_2state(y, seed=42)
        country_results[cc] = res
        print(f"  {COUNTRY_LABEL[cc]:12s}: stress={res.n_stress}/{res.n_total} "
              f"({100*res.n_stress/res.n_total:.0f}%)  "
              f"RV calm={np.exp(res.mu_calm):.3f} stress={np.exp(res.mu_stress):.3f}  "
              f"dur_stress={res.dur_stress:.1f}m")
        stats_rows.append({
            "regime_def":"COUNTRY","unit":cc,
            "switching_var":res.switching_variance,"llf":round(res.llf,2),
            "mu_calm":round(res.mu_calm,3),"mu_stress":round(res.mu_stress,3),
            "rv_calm":round(np.exp(res.mu_calm),3),"rv_stress":round(np.exp(res.mu_stress),3),
            "n_stress":res.n_stress,"n_total":res.n_total,
            "pct_stress":round(100*res.n_stress/res.n_total,1),
            "p_stay_calm":round(res.p_stay_calm,3),"p_stay_stress":round(res.p_stay_stress,3),
            "dur_calm":round(res.dur_calm,1),"dur_stress":round(res.dur_stress,1),
        })
    except Exception as e:
        print(f"  {cc}: FAILED — {e}")

# ══════════════════════════════════════════════════════════════════════════════
# C. MERGE regime flags back into panel
# ══════════════════════════════════════════════════════════════════════════════
panel["stress_common"]  = panel.index.map(common_stress.to_dict()).astype(float)
panel["pstress_common"] = panel.index.map(common_pstress.to_dict())

# country-specific: map per (country, date)
panel["stress_country"]  = np.nan
panel["pstress_country"] = np.nan
for cc, res in country_results.items():
    mask = panel["country"] == cc
    idx  = panel.index[mask]
    panel.loc[mask, "stress_country"]  = idx.map(res.stress_flag.to_dict())
    panel.loc[mask, "pstress_country"] = idx.map(res.smoothed_stress.to_dict())

# Agreement between common and country regimes
agree = (panel["stress_common"] == panel["stress_country"]).mean()
print(f"\n  Regime agreement (common vs country): {100*agree:.1f}%")

panel.reset_index().to_parquet(OUT_PANEL, index=False)
print(f"\nSaved panel with regimes: {OUT_PANEL}")

df_stats = pd.DataFrame(stats_rows)
df_stats.to_csv(TABLES_DIR / "tableMS1_regime_stats.csv", index=False)
print(f"Saved: tableMS1_regime_stats.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: regime probabilities
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[1.1, 1])

# Top: common regime
ax0 = axes[0]
ax0.fill_between(common_pstress.index, 0, common_pstress.values,
                 color="#d62728", alpha=0.4, label="P(Stress regime)")
ax0.plot(regional.index, (regional - regional.min())/(regional.max()-regional.min()),
         color="#333", lw=1, alpha=0.6, label="Regional log_rv (scaled)")
ax0.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
for t0, t1, lab in CRISES:
    ax0.axvspan(pd.Timestamp(t0), pd.Timestamp(t1), color="grey", alpha=0.15)
    ax0.text(pd.Timestamp(t0), 1.02, lab, fontsize=7, rotation=0, color="#555")
ax0.set_ylim(0, 1.1)
ax0.set_ylabel("P(Stress)", fontsize=9)
ax0.set_title("A. Common ASEAN Volatility Regime (Markov-Switching)\n"
              f"Stress = {res_common.n_stress}/{res_common.n_total} months "
              f"({100*res_common.n_stress/res_common.n_total:.0f}%)  |  "
              f"Expected stress duration = {res_common.dur_stress:.1f} months",
              fontsize=10, fontweight="bold")
ax0.legend(fontsize=8, loc="upper right")
ax0.grid(alpha=0.2)

# Bottom: country-specific P(stress) heatmap-style
ax1 = axes[1]
for i, cc in enumerate(CC5):
    if cc not in country_results:
        continue
    sp = country_results[cc].smoothed_stress
    ax1.plot(sp.index, sp.values + i*1.1, color=CC_COLORS[cc], lw=1.2,
             label=COUNTRY_LABEL[cc])
    ax1.axhline(i*1.1 + 0.5, color=CC_COLORS[cc], ls=":", lw=0.5, alpha=0.4)
ax1.set_yticks([i*1.1 + 0.5 for i in range(len(CC5))])
ax1.set_yticklabels([COUNTRY_LABEL[cc] for cc in CC5], fontsize=8)
for t0, t1, lab in CRISES:
    ax1.axvspan(pd.Timestamp(t0), pd.Timestamp(t1), color="grey", alpha=0.12)
ax1.set_title("B. Country-Specific Stress Probabilities (stacked)", fontsize=10, fontweight="bold")
ax1.set_xlabel("Date", fontsize=9)
ax1.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(FIGS_DIR / "figMS1_regime_probabilities.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGS_DIR / "figMS1_regime_probabilities.pdf", bbox_inches="tight")  # vector PDF
plt.close()
print(f"Saved: figMS1_regime_probabilities.png")

print("\nMS1 HOAN THANH.")
