"""MS10: Does GPR predict the ONSET of a stress regime? (new contribution)

Beyond "GPR amplifies volatility within stress" (b3), we ask a second channel:
  Does high GPR raise the PROBABILITY of transitioning INTO a stress regime?

Two complementary approaches:
  A) Pooled probit / logit:
     P(Stress_{i,t+1}=1 | calm at t) = F(a + g1*GPRC_t + g2*logRV_t + g3*VIX_t + ...)
     -> marginal effect of GPR on stress-onset probability
  B) Transition-conditioned: among CALM months only, does GPR predict switching
     to stress next month? (clean onset test)

This adds a second transmission mechanism: GPR -> regime onset (not just within-regime amplification).

Outputs:
  results/tables/tableMS10_tvtp.csv
  results/figures/figMS10_gpr_regime_onset.png
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
from statsmodels.discrete.discrete_model import Logit
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
panel["log_rv"]   = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_t"] = panel["log_rv"]

# Next-month stress (onset target) per country
panel["stress_next"] = panel.groupby("country")[STRESS].shift(-1)

# Standardize predictors within country
for col in ["gpr_country","vix_m","dlog_fx_m","log_rv_t","sp500_ret_m","us10y_m","wti_ret_m"]:
    panel[col+"_z"] = panel.groupby("country")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))

PREDICT = ["gpr_country_z","log_rv_t_z","vix_m_z","dlog_fx_m_z",
           "sp500_ret_m_z","us10y_m_z","wti_ret_m_z"]

rows = []

# ══════════════════════════════════════════════════════════════════════════════
# A. POOLED LOGIT — all months
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("MS10: GPR -> STRESS REGIME ONSET (TVTP via logit)")
print("="*65)

def run_logit(df, predictors, label, cluster_time=True):
    d = df.dropna(subset=["stress_next"]+predictors).copy()
    if len(d) < 60 or d["stress_next"].nunique() < 2:
        return None
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[predictors].reset_index(drop=True),
                                   cty.reset_index(drop=True)], axis=1).astype(float))
    names = ["const"]+predictors+list(cty.columns)
    try:
        if cluster_time:
            tg = pd.factorize(d.index)[0]
            m = Logit(d["stress_next"].values, X.values).fit(disp=0,
                      cov_type="cluster", cov_kwds={"groups": tg})
        else:
            m = Logit(d["stress_next"].values, X.values).fit(disp=0)
    except Exception as e:
        print(f"   {label}: {e}"); return None
    i = names.index("gpr_country_z")
    b, se, p = m.params[i], m.bse[i], m.pvalues[i]
    # average marginal effect of GPR (approx): mean p*(1-p)*beta
    phat = m.predict(X.values)
    ame = np.mean(phat*(1-phat)) * b
    return {"label":label,"gpr_coef":round(b,4),"se":round(se,4),"p_value":round(p,4),
            "ame_pct":round(ame*100,3),"n":len(d),"n_onset":int(d["stress_next"].sum()),
            "sig":"***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""}

# A1: all months
r = run_logit(panel, PREDICT, "All months (pooled)")
if r:
    rows.append({**r,"sample":"all"})
    print(f"\n  [A1] All months: GPR coef={r['gpr_coef']:+.4f} (p={r['p_value']:.3f}) {r['sig']}")
    print(f"       AME: +1 SD GPR -> {r['ame_pct']:+.2f} pp stress-onset prob  (n={r['n']}, onsets={r['n_onset']})")

# A2: GPR-only (no other predictors) for raw association
r2 = run_logit(panel, ["gpr_country_z"], "GPR only")
if r2:
    rows.append({**r2,"sample":"all_gpronly"})
    print(f"  [A2] GPR-only: coef={r2['gpr_coef']:+.4f} (p={r2['p_value']:.3f}) {r2['sig']}")

# ══════════════════════════════════════════════════════════════════════════════
# B. ONSET TEST — among CALM months only (clean transition into stress)
# ══════════════════════════════════════════════════════════════════════════════
calm = panel[panel[STRESS] == 0]
r3 = run_logit(calm, PREDICT, "Calm months only (onset)")
if r3:
    rows.append({**r3,"sample":"calm_only"})
    print(f"\n  [B] Onset from CALM: GPR coef={r3['gpr_coef']:+.4f} (p={r3['p_value']:.3f}) {r3['sig']}")
    print(f"      AME: +1 SD GPR -> {r3['ame_pct']:+.2f} pp prob of entering stress next month")
    print(f"      (n_calm={r3['n']}, transitions to stress={r3['n_onset']})")

# ══════════════════════════════════════════════════════════════════════════════
# C. PER-COUNTRY onset
# ══════════════════════════════════════════════════════════════════════════════
print("\n  [C] Per-country onset (GPR coef in logit):")
for cc in CC5:
    sub = panel[panel["country"]==cc]
    rc = run_logit(sub, ["gpr_country_z","log_rv_t_z","vix_m_z"], cc, cluster_time=False)
    if rc:
        rows.append({**rc,"sample":f"country_{cc}"})
        print(f"      {COUNTRY_LABEL[cc]:12s}: GPR coef={rc['gpr_coef']:+.4f} (p={rc['p_value']:.3f}) {rc['sig']}  onsets={rc['n_onset']}")

df = pd.DataFrame(rows)
df.to_csv(TABLES_DIR / "tableMS10_tvtp.csv", index=False)
print(f"\nSaved: tableMS10_tvtp.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: predicted stress-onset prob vs GPR level
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: GPR coef across samples
ax0 = axes[0]
plot_df = df[df["sample"].isin(["all","calm_only","all_gpronly"])].copy()
labmap = {"all":"All months\n(full controls)","calm_only":"Calm only\n(onset)",
          "all_gpronly":"GPR only\n(raw)"}
plot_df["plot_label"] = plot_df["sample"].map(labmap)
cols = ["#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0" for p in plot_df["p_value"]]
ax0.bar(plot_df["plot_label"], plot_df["gpr_coef"], yerr=1.96*plot_df["se"],
        color=cols, alpha=0.85, capsize=6)
for i,(v,p) in enumerate(zip(plot_df["gpr_coef"], plot_df["p_value"])):
    s = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "n.s."
    ax0.text(i, v+0.02, s, ha="center", fontsize=10, fontweight="bold")
ax0.axhline(0, color="k", lw=0.8)
ax0.set_ylabel("GPR coefficient (logit, stress onset)", fontsize=9)
ax0.set_title("Does GPR predict stress-regime onset?\n(logit, time-clustered SE)",
              fontsize=10, fontweight="bold")
ax0.grid(axis="y", alpha=0.3)

# Right: predicted onset probability vs GPR (from calm-only model)
ax1 = axes[1]
calm_d = calm.dropna(subset=["stress_next","gpr_country_z","log_rv_t_z","vix_m_z"]).copy()
if len(calm_d) > 50:
    # bin GPR into quintiles, plot empirical onset rate
    calm_d["gpr_bin"] = pd.qcut(calm_d["gpr_country_z"], 5, labels=False, duplicates="drop")
    onset_rate = calm_d.groupby("gpr_bin")["stress_next"].mean()
    gpr_mean   = calm_d.groupby("gpr_bin")["gpr_country_z"].mean()
    ax1.plot(gpr_mean.values, onset_rate.values*100, "o-", color="#d62728", lw=2, ms=8)
    ax1.set_xlabel("GPR_country (standardized, quintile mean)", fontsize=9)
    ax1.set_ylabel("Stress-onset rate next month (%)", fontsize=9)
    ax1.set_title("Empirical Stress-Onset Rate by GPR Level\n(among calm months)",
                  fontsize=10, fontweight="bold")
    ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS_DIR / "figMS10_gpr_regime_onset.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figMS10_gpr_regime_onset.png")
print("\nMS10 HOAN THANH.")
