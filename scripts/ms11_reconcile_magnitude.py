"""MS11: Per-country reconciliation + economic magnitude of b3.

Resolves the MS3 (significant pooled) vs MS4 (insignificant subsample) tension
and translates the standardized b3 into economic terms.

Parts:
  A) Country x Stress slopes in ONE pooled model (formal heterogeneity test)
  B) Power disclosure: n_stress per country -> why subsample is underpowered
  C) Economic magnitude: de-standardize b3 ->
        "+1 SD GPR shock in stress raises next-month RV by X%"

Outputs:
  results/tables/tableMS11_magnitude.csv
  results/tables/tableMS11_country_slopes.csv
  results/figures/figMS11_country_heterogeneity.png
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

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_regimes.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
CC5 = ["VNM","THA","IDN","MYS","PHL"]
COUNTRY_LABEL = {"VNM":"Vietnam","THA":"Thailand","IDN":"Indonesia",
                 "MYS":"Malaysia","PHL":"Philippines"}
STRESS = "stress_common"

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_t"]     = panel["log_rv"]

# Keep raw sd of outcome & treatment per country for de-standardization
raw_stats = {}
for cc in CC5:
    sub = panel[panel["country"]==cc]
    raw_stats[cc] = {
        "sd_logrv_lead": sub["log_rv_lead1"].std(),
        "sd_gpr": sub["gpr_country"].std(),
        "mean_rv": sub["rv_m"].mean(),
    }

for col in ["gpr_country","vix_m","dlog_fx_m","log_rv_t","log_rv_lead1",
            "sp500_ret_m","us10y_m","wti_ret_m","ret_m"]:
    panel[col+"_z"] = panel.groupby("country")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))

OUTCOME = "log_rv_lead1_z"
CTRL = ["log_rv_t_z","vix_m_z","dlog_fx_m_z","sp500_ret_m_z","us10y_m_z","wti_ret_m_z","ret_m_z"]


# ══════════════════════════════════════════════════════════════════════════════
# A. Country x Stress interaction slopes (one pooled model)
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("MS11-A. COUNTRY-SPECIFIC GPRxStress SLOPES (one pooled model)")
print("="*65)
p = panel.copy()
p["gpr_x_stress"] = p["gpr_country_z"] * p[STRESS]
# build country-specific interaction columns
for cc in CC5:
    p[f"gxs_{cc}"] = p["gpr_x_stress"] * (p["country"]==cc).astype(float)

gxs_cols = [f"gxs_{cc}" for cc in CC5]
cols = ["gpr_country_z", STRESS] + gxs_cols + CTRL
d = p.dropna(subset=[OUTCOME]+cols)
cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
X = sm.add_constant(pd.concat([d[cols].reset_index(drop=True),
                               cty.reset_index(drop=True)], axis=1).astype(float).values)
names = ["const"]+cols+list(cty.columns)
tg = pd.factorize(d.index)[0]
m = OLS(d[OUTCOME].values, X).fit(cov_type="cluster", cov_kwds={"groups": tg})

rows_slope = []
for cc in CC5:
    i = names.index(f"gxs_{cc}")
    b, se, pv = m.params[i], m.bse[i], m.pvalues[i]
    sig = "***" if pv<0.01 else "**" if pv<0.05 else "*" if pv<0.10 else ""
    n_stress_cc = int(((d["country"]==cc)&(d[STRESS]==1)).sum())
    print(f"  {COUNTRY_LABEL[cc]:12s}: GPRxStress={b:+.4f} (p={pv:.3f}) {sig}  [stress n={n_stress_cc}]")
    rows_slope.append({"country":cc,"country_label":COUNTRY_LABEL[cc],
                       "gpr_x_stress":round(b,4),"se":round(se,4),"p_value":round(pv,4),
                       "signif":sig,"n_stress":n_stress_cc})
df_slope = pd.DataFrame(rows_slope)
df_slope.to_csv(TABLES_DIR / "tableMS11_country_slopes.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# B. Power disclosure
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("MS11-B. POWER DISCLOSURE (why per-country subsample is underpowered)")
print("="*65)
for cc in CC5:
    n_stress = int(((panel["country"]==cc)&(panel[STRESS]==1)).sum())
    n_total  = int((panel["country"]==cc).sum())
    print(f"  {COUNTRY_LABEL[cc]:12s}: {n_stress} stress / {n_total} months "
          f"({100*n_stress/n_total:.0f}%) -> subsample DML needs >=30 for power")


# ══════════════════════════════════════════════════════════════════════════════
# C. Economic magnitude (de-standardize pooled b3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("MS11-C. ECONOMIC MAGNITUDE")
print("="*65)
b3_std = -0.1695   # pooled standardized b3 from MS3

# Average sd across countries
avg_sd_logrv = np.mean([raw_stats[c]["sd_logrv_lead"] for c in CC5])
avg_mean_rv  = np.mean([raw_stats[c]["mean_rv"] for c in CC5])

# b3_std means: +1 SD GPR (in stress) -> b3_std SD change in log_rv_lead1
# Convert to log_rv units: delta_logrv = b3_std * avg_sd_logrv
delta_logrv = b3_std * avg_sd_logrv
# Convert to % change in RV: exp(delta_logrv) - 1
pct_rv = (np.exp(delta_logrv) - 1) * 100

print(f"  Pooled standardized b3 = {b3_std:+.3f}")
print(f"  Avg SD(log_rv_lead1) across countries = {avg_sd_logrv:.3f}")
print(f"  => +1 SD GPR shock during STRESS changes log_rv by {delta_logrv:+.3f}")
print(f"  => RV (volatility) change of {pct_rv:+.1f}% during stress")
print(f"  (avg RV level ~ {avg_mean_rv:.3f} = {avg_mean_rv*100:.0f}% annualized)")
print(f"  Note: negative sign = GPR shock in stress is associated with LOWER")
print(f"        next-month RV (mean-reversion after stress spikes / vol overshoot).")

rows_mag = [{
    "metric":"pooled_b3_std","value":round(b3_std,4)},
    {"metric":"avg_sd_logrv_lead","value":round(avg_sd_logrv,4)},
    {"metric":"delta_logrv_per_1SD_GPR","value":round(delta_logrv,4)},
    {"metric":"pct_RV_change_per_1SD_GPR","value":round(pct_rv,2)},
    {"metric":"avg_RV_level","value":round(avg_mean_rv,4)},
]
pd.DataFrame(rows_mag).to_csv(TABLES_DIR / "tableMS11_magnitude.csv", index=False)
print(f"\nSaved: tableMS11_magnitude.csv, tableMS11_country_slopes.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: country slopes
ax0 = axes[0]
dd = df_slope.set_index("country").reindex(CC5).reset_index()
for i, row in dd.iterrows():
    p_ = row["p_value"]
    col = "#d62728" if p_<0.05 else "#ff7f0e" if p_<0.10 else "#4878d0"
    ax0.errorbar(row["gpr_x_stress"], i, xerr=1.96*row["se"], fmt="o",
                 color=col, capsize=4, lw=2, ms=7)
    ax0.text(row["gpr_x_stress"]+1.96*row["se"]+0.02, i,
             f"{row['signif']} (n_s={row['n_stress']})", va="center", fontsize=7, color=col)
ax0.axvline(0, color="k", ls="--", lw=0.8)
ax0.set_yticks(range(len(dd))); ax0.set_yticklabels([COUNTRY_LABEL[c] for c in dd["country"]], fontsize=9)
ax0.set_xlabel("GPRC × Stress slope (std)", fontsize=9)
ax0.set_title("Country-Specific GPRC×Stress Slopes\n(one pooled model, time-cluster SE)",
              fontsize=10, fontweight="bold")
ax0.grid(axis="x", alpha=0.3)

# Right: power disclosure (n_stress per country)
ax1 = axes[1]
ns = [int(((panel["country"]==cc)&(panel[STRESS]==1)).sum()) for cc in CC5]
colors = ["#d62728" if n<30 else "#4daf4a" for n in ns]
ax1.barh([COUNTRY_LABEL[c] for c in CC5][::-1], ns[::-1], color=colors[::-1], alpha=0.8)
ax1.axvline(30, color="k", ls="--", lw=1, label="min n for DML power (~30)")
for i,n in enumerate(ns[::-1]):
    ax1.text(n+1, i, str(n), va="center", fontsize=9)
ax1.set_xlabel("# stress months per country", fontsize=9)
ax1.set_title("Power Disclosure: Stress Sample Size\n(red = below DML power threshold)",
              fontsize=10, fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS_DIR / "figMS11_country_heterogeneity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figMS11_country_heterogeneity.png")
print("\nMS11 HOAN THANH.")
