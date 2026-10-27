"""MS9: Placebo / Falsification tests for the GPRC x Stress result.

Three placebos to prove the regime carries real information:
  1. RANDOM REGIME SHUFFLE: permute stress labels 1000x -> distribution of
     placebo b3. The true b3 must lie in the tail (empirical p < 0.05).
  2. RETURN PLACEBO: GPRC x Stress -> return_{t+1} (not RV). Should be weak
     (confirms risk/volatility channel, not a mean-return/alpha channel).
  3. PHASE-SHIFT PLACEBO: shift stress labels by +/-3, +/-6 months. The effect
     should weaken if it depends on the correct timing of stress.

Outputs:
  results/tables/tableMS9_placebo.csv
  results/figures/figMS9_placebo_distribution.png
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
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from src.utils.io import ROOT

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_regimes.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
CC5 = ["VNM","THA","IDN","MYS","PHL"]
STRESS = "stress_common"

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["ret_lead1"]    = panel.groupby("country")["ret_m"].shift(-1)
panel["log_rv_t"]     = panel["log_rv"]

for col in ["gpr_country","vix_m","dlog_fx_m","log_rv_t","log_rv_lead1",
            "ret_lead1","sp500_ret_m","us10y_m","wti_ret_m","ret_m"]:
    panel[col+"_z"] = panel.groupby("country")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))

CTRL = ["gpr_country_z","log_rv_t_z","vix_m_z","dlog_fx_m_z",
        "sp500_ret_m_z","us10y_m_z","wti_ret_m_z","ret_m_z"]

def b3_for_stress(stress_series, outcome="log_rv_lead1_z"):
    """Fast b3 via OLS (no SE) given a date-indexed stress series."""
    p = panel.copy()
    p["stress_p"] = p.index.map(pd.Series(stress_series).to_dict())
    p["gpr_x_stress_p"] = p["gpr_country_z"] * p["stress_p"]
    cols = ["stress_p","gpr_x_stress_p"] + CTRL
    d = p.dropna(subset=[outcome]+cols)
    if len(d) < 80 or d["stress_p"].nunique()<2:
        return np.nan
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[cols].reset_index(drop=True),
                                   cty.reset_index(drop=True)], axis=1).astype(float).values)
    beta = np.linalg.lstsq(X, d[outcome].values, rcond=None)[0]
    return beta[2]   # const, stress_p, gpr_x_stress_p

def b3_with_p(stress_series, outcome="log_rv_lead1_z"):
    p = panel.copy()
    p["stress_p"] = p.index.map(pd.Series(stress_series).to_dict())
    p["gpr_x_stress_p"] = p["gpr_country_z"] * p["stress_p"]
    cols = ["stress_p","gpr_x_stress_p"] + CTRL
    d = p.dropna(subset=[outcome]+cols)
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[cols].reset_index(drop=True),
                                   cty.reset_index(drop=True)], axis=1).astype(float).values)
    tg = pd.factorize(d.index)[0]
    m = OLS(d[outcome].values, X).fit(cov_type="cluster", cov_kwds={"groups": tg})
    return m.params[2], m.bse[2], m.pvalues[2], len(d)

# Common stress series (date -> 0/1)
dates = panel.index.unique()
stress_common = panel.groupby(panel.index)[STRESS].first().reindex(dates)

# True b3
b3_true, se_true, p_true, n_true = b3_with_p(stress_common)
print("="*65)
print("MS9: PLACEBO / FALSIFICATION TESTS")
print("="*65)
print(f"\n  TRUE b3 (GPRC x Stress -> logRV): {b3_true:+.4f} (p={p_true:.3f}, n={n_true})")

rows = []

# ── 1. Random regime shuffle ──────────────────────────────────────────────────
print("\n  [1] RANDOM REGIME SHUFFLE (1000 permutations)...")
n_perm = 1000
rng = np.random.RandomState(42)
stress_vals = stress_common.values.copy()
placebo_b3 = np.full(n_perm, np.nan)
for i in range(n_perm):
    perm = rng.permutation(stress_vals)
    s = pd.Series(perm, index=dates)
    placebo_b3[i] = b3_for_stress(s)
placebo_b3 = placebo_b3[~np.isnan(placebo_b3)]
# two-sided empirical p: fraction of |placebo| >= |true|
emp_p = np.mean(np.abs(placebo_b3) >= abs(b3_true))
pct_more_neg = np.mean(placebo_b3 <= b3_true)
print(f"      Placebo b3: mean={placebo_b3.mean():+.4f}, sd={placebo_b3.std():.4f}")
print(f"      True b3={b3_true:+.4f}  -> empirical p={emp_p:.4f} "
      f"(fraction <= true: {pct_more_neg:.4f})")
print(f"      {'PASS: true effect in tail' if emp_p<0.05 else 'WEAK'}")
rows.append({"placebo":"Random shuffle (1000x)","b3_true":round(b3_true,4),
             "placebo_mean":round(placebo_b3.mean(),4),"placebo_sd":round(placebo_b3.std(),4),
             "empirical_p":round(emp_p,4),"verdict":"PASS" if emp_p<0.05 else "WEAK"})

# ── 2. Return placebo ─────────────────────────────────────────────────────────
print("\n  [2] RETURN PLACEBO (GPRC x Stress -> return_{t+1})...")
b3_ret, se_ret, p_ret, n_ret = b3_with_p(stress_common, outcome="ret_lead1_z")
print(f"      b3 on RETURN: {b3_ret:+.4f} (p={p_ret:.3f})")
print(f"      b3 on RV:     {b3_true:+.4f} (p={p_true:.3f})")
print(f"      {'PASS: risk channel (RV strong, return weak)' if p_ret>0.10 else 'return also significant'}")
rows.append({"placebo":"Return placebo","b3_true":round(b3_ret,4),
             "placebo_mean":np.nan,"placebo_sd":round(se_ret,4),
             "empirical_p":round(p_ret,4),
             "verdict":"PASS (risk channel)" if p_ret>0.10 else "return also sig"})

# ── 3. Phase-shift placebo ────────────────────────────────────────────────────
print("\n  [3] PHASE-SHIFT PLACEBO (shift stress labels)...")
for shift in [-6, -3, 3, 6]:
    s_shift = pd.Series(stress_common.values, index=dates).shift(shift)
    try:
        b3_s, se_s, p_s, n_s = b3_with_p(s_shift.dropna())
        print(f"      shift {shift:+d}m: b3={b3_s:+.4f} (p={p_s:.3f})  "
              f"[{'weakened' if abs(b3_s)<abs(b3_true) else 'not weakened'}]")
        rows.append({"placebo":f"Phase-shift {shift:+d}m","b3_true":round(b3_s,4),
                     "placebo_mean":np.nan,"placebo_sd":round(se_s,4),
                     "empirical_p":round(p_s,4),
                     "verdict":"weakened" if abs(b3_s)<abs(b3_true) else "not weakened"})
    except Exception as e:
        print(f"      shift {shift}: {e}")

df = pd.DataFrame(rows)
df.to_csv(TABLES_DIR / "tableMS9_placebo.csv", index=False)
print(f"\nSaved: tableMS9_placebo.csv")


# ── FIGURE ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax0 = axes[0]
ax0.hist(placebo_b3, bins=45, color="#bbb", alpha=0.85, label="Placebo b3 (shuffled regime)")
ax0.axvline(b3_true, color="#d62728", lw=2.5, label=f"True b3={b3_true:.3f}")
ax0.axvline(np.percentile(placebo_b3,2.5), color="k", ls=":", lw=1)
ax0.axvline(np.percentile(placebo_b3,97.5), color="k", ls=":", lw=1, label="95% placebo band")
ax0.set_xlabel("b3 (GPRC × Stress)", fontsize=9)
ax0.set_title(f"[1] Random Regime Shuffle (1000x)\n"
              f"True effect empirical p={emp_p:.3f}", fontsize=10, fontweight="bold")
ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

ax1 = axes[1]
labels = ["RV\n(true outcome)","Return\n(placebo)"]
vals   = [b3_true, b3_ret]
ses    = [se_true, se_ret]
ps     = [p_true, p_ret]
cols   = ["#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0" for p in ps]
ax1.bar(labels, vals, yerr=[1.96*s for s in ses], color=cols, alpha=0.85, capsize=6)
for i,(v,p) in enumerate(zip(vals,ps)):
    s = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "n.s."
    ax1.text(i, v-0.012, s, ha="center", fontsize=11, fontweight="bold")
ax1.axhline(0, color="k", lw=0.8)
ax1.set_ylabel("b3 (GPRC × Stress)", fontsize=9)
ax1.set_title("[2] Return Placebo\nRisk channel: strong for RV, weak for return",
              fontsize=10, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS_DIR / "figMS9_placebo_distribution.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGS_DIR / "figMS9_placebo_distribution.pdf", bbox_inches="tight")  # vector PDF
plt.close()
print(f"Saved: figMS9_placebo_distribution.png")
print("\nMS9 HOAN THANH.")
