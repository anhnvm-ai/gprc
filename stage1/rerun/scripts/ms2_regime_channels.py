"""MS2: Regime-Specific Channel Ranking (Step 2 of upgrade guide)

Estimate volatility channels SEPARATELY for Calm vs Stress regimes.
Standardized FE regression (country FE) within each regime.

Model (per regime):
  logRV_{i,t+1} = a_i + th1*logRV_{i,t} + th2*VIX + th3*dlogFX + th4*GPRC + ctrls

Tests:
  H1: persistence dominant in both regimes
  H2: persistence stronger in stress
  H3: VIX stronger in stress
  H4: GPRC matters mainly in stress

Outputs:
  results/tables/tableMS2_regime_channels.csv
  results/figures/figMS2_regime_channels.png
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
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CC5 = ["VNM","THA","IDN","MYS","PHL"]

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()

panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_t"]     = panel["log_rv"]   # contemporaneous persistence predictor

CHANNELS = {
    "log_rv_t":    "Volatility Persistence",
    "vix_m":       "VIX (Global Risk)",
    "dlog_fx_m":   "FX Shock",
    "gpr_country": "GPR Country",
}
CONTROLS = ["sp500_ret_m","us10y_m","wti_ret_m","ret_m"]
OUTCOME  = "log_rv_lead1"

# Standardize within country (full sample, so calm/stress comparable)
for col in list(CHANNELS) + CONTROLS + [OUTCOME]:
    if col in panel.columns:
        panel[col+"_z"] = panel.groupby("country")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12))

ch_z   = [c+"_z" for c in CHANNELS]
ctrl_z = [c+"_z" for c in CONTROLS if c+"_z" in panel.columns]
y_col  = OUTCOME+"_z"


def fe_regression(df, regime_label):
    """Standardized FE OLS (country dummies) clustered by country."""
    df = df.dropna(subset=[y_col] + ch_z + ctrl_z).copy()
    n = len(df)
    if n < 40:
        return None, n
    cty_d = pd.get_dummies(df["country"], prefix="cty", drop_first=True).astype(float)
    X = pd.concat([df[ch_z + ctrl_z].reset_index(drop=True),
                   cty_d.reset_index(drop=True)], axis=1).astype(float)
    Xc = sm.add_constant(X.values)
    m = OLS(df[y_col].values, Xc).fit(cov_type="cluster",
                                      cov_kwds={"groups": df["country"].values})
    rows = []
    for i, (col, label) in enumerate(CHANNELS.items()):
        b = m.params[i+1]; se = m.bse[i+1]; p = m.pvalues[i+1]
        sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        rows.append({"regime":regime_label,"channel":col,"channel_label":label,
                     "beta_std":round(b,4),"se":round(se,4),"p_value":round(p,4),
                     "signif":sig,"abs_beta":abs(b),"n_obs":n,"r2":round(m.rsquared,4)})
    return pd.DataFrame(rows), n


# ── Run for COMMON regime ─────────────────────────────────────────────────────
print("="*65)
print("MS2: REGIME-SPECIFIC CHANNELS (Common ASEAN regime)")
print("="*65)

all_rows = []
for regime_val, regime_label in [(0,"Calm"), (1,"Stress")]:
    sub = panel[panel["stress_common"] == regime_val]
    df_reg, n = fe_regression(sub, regime_label)
    if df_reg is None:
        print(f"\n  {regime_label}: n={n} too few")
        continue
    print(f"\n  --- {regime_label} regime (n={n}, R2={df_reg['r2'].iloc[0]}) ---")
    df_sorted = df_reg.sort_values("abs_beta", ascending=False).reset_index(drop=True)
    for rank, (_, r) in enumerate(df_sorted.iterrows(), 1):
        print(f"    Rank {rank}: {r['channel_label']:25s} beta={r['beta_std']:+.4f}  p={r['p_value']:.3f} {r['signif']}")
    all_rows.append(df_reg)

df_common = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
df_common["regime_def"] = "COMMON"


# ── Run for COUNTRY-specific regime ───────────────────────────────────────────
print("\n" + "="*65)
print("MS2b: REGIME-SPECIFIC CHANNELS (Country-specific regime)")
print("="*65)
all_rows_cty = []
for regime_val, regime_label in [(0,"Calm"), (1,"Stress")]:
    sub = panel[panel["stress_country"] == regime_val]
    df_reg, n = fe_regression(sub, regime_label)
    if df_reg is None:
        print(f"  {regime_label}: n={n} too few"); continue
    print(f"  {regime_label} (n={n}): " + "  ".join(
        f"{r['channel_label'].split()[0]}={r['beta_std']:+.3f}{r['signif']}"
        for _, r in df_reg.iterrows()))
    all_rows_cty.append(df_reg)
df_cty = pd.concat(all_rows_cty, ignore_index=True) if all_rows_cty else pd.DataFrame()
df_cty["regime_def"] = "COUNTRY"

df_out = pd.concat([df_common, df_cty], ignore_index=True)
df_out.to_csv(TABLES_DIR / "tableMS2_regime_channels.csv", index=False)
print(f"\nSaved: tableMS2_regime_channels.csv")


# ── Hypothesis checks (common regime) ─────────────────────────────────────────
print("\n" + "="*65)
print("HYPOTHESIS CHECKS (Common regime)")
print("="*65)
def get_beta(df, regime, channel):
    r = df[(df["regime"]==regime) & (df["channel"]==channel)]
    return (float(r["beta_std"].iloc[0]), float(r["p_value"].iloc[0])) if len(r) else (np.nan, np.nan)

if not df_common.empty:
    p_calm, _   = get_beta(df_common,"Calm","log_rv_t")
    p_stress, _ = get_beta(df_common,"Stress","log_rv_t")
    v_calm, _   = get_beta(df_common,"Calm","vix_m")
    v_stress, _ = get_beta(df_common,"Stress","vix_m")
    g_calm, gpc = get_beta(df_common,"Calm","gpr_country")
    g_stress, gps = get_beta(df_common,"Stress","gpr_country")
    print(f"  H2 (persistence stronger in stress): |{p_stress:.3f}| vs |{p_calm:.3f}| "
          f"-> {'SUPPORTED' if abs(p_stress)>abs(p_calm) else 'NOT supported'}")
    print(f"  H3 (VIX stronger in stress):         |{v_stress:.3f}| vs |{v_calm:.3f}| "
          f"-> {'SUPPORTED' if abs(v_stress)>abs(v_calm) else 'NOT supported'}")
    print(f"  H4 (GPRC matters mainly in stress):  calm beta={g_calm:+.3f}(p={gpc:.2f})  "
          f"stress beta={g_stress:+.3f}(p={gps:.2f}) "
          f"-> {'SUPPORTED' if (gps<0.10 and gpc>=0.10) else 'mixed/NOT'}")


# ── FIGURE ────────────────────────────────────────────────────────────────────
if not df_common.empty:
    ch_order = list(CHANNELS.keys())
    labels   = [CHANNELS[c] for c in ch_order]
    calm   = [get_beta(df_common,"Calm",c)[0]   for c in ch_order]
    stress = [get_beta(df_common,"Stress",c)[0] for c in ch_order]
    calm_p   = [get_beta(df_common,"Calm",c)[1]   for c in ch_order]
    stress_p = [get_beta(df_common,"Stress",c)[1] for c in ch_order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(ch_order)); h = 0.35
    b1 = ax.barh(y+h/2, calm,   h, color="#4daf4a", alpha=0.8, label="Calm regime")
    b2 = ax.barh(y-h/2, stress, h, color="#d62728", alpha=0.8, label="Stress regime")
    for yi, (v, p) in zip(y+h/2, zip(calm, calm_p)):
        s = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        if s: ax.text(v + (0.01 if v>=0 else -0.01), yi, s, va="center",
                      ha="left" if v>=0 else "right", fontsize=9)
    for yi, (v, p) in zip(y-h/2, zip(stress, stress_p)):
        s = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        if s: ax.text(v + (0.01 if v>=0 else -0.01), yi, s, va="center",
                      ha="left" if v>=0 else "right", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Standardized β  (logRV_{t+1})", fontsize=9)
    ax.set_title("Regime-Specific Channel Effects — 5-ASEAN\n"
                 "Calm vs Stress (Common ASEAN regime) | Country FE | "
                 "* p<0.10  ** p<0.05  *** p<0.01",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "figMS2_regime_channels.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figMS2_regime_channels.png")

print("\nMS2 HOAN THANH.")
