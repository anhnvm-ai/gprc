"""MS4: Regime-Specific DML Orthogonal Estimation (Step 4 of upgrade guide)

Estimate the orthogonal causal effect of each channel SEPARATELY within
Calm and Stress regimes, with all other channels (incl. persistence) partialled
out. This is the cleanest test: does GPRC have an independent effect in stress
after benchmarking against the dominant persistence + VIX channels?

For each regime r in {Calm, Stress}:
  For each channel D in {logRV_t, VIX, dlogFX, GPRC}:
    DML(Y=logRV_{t+1}, D, X = other channels + macro + country FE)

Outputs:
  results/tables/tableMS4_regime_dml.csv
  results/figures/figMS4_regime_dml.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils.io import ROOT
from src.models.dml import DMLEstimator

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_regimes.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CC5 = ["VNM","THA","IDN","MYS","PHL"]
STRESS = "stress_common"

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_t"]     = panel["log_rv"]

CHANNELS = {
    "log_rv_t":    "Volatility Persistence",
    "vix_m":       "VIX (Global Risk)",
    "dlog_fx_m":   "FX Shock",
    "gpr_country": "GPR Country",
}
MACRO = ["sp500_ret_m","us10y_m","wti_ret_m","ret_m"]
OUTCOME = "log_rv_lead1"

for col in list(CHANNELS) + MACRO + [OUTCOME]:
    panel[col+"_z"] = panel.groupby("country")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))

ch_z    = {k:k+"_z" for k in CHANNELS}
macro_z = [c+"_z" for c in MACRO]
y_col   = OUTCOME+"_z"


def dml_pooled(df, treat_z, other_z):
    ctrl = other_z + macro_z
    d = df.dropna(subset=[y_col, treat_z]+ctrl).copy()
    if len(d) < 40:
        return None
    cty_d = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    X = pd.concat([d[ctrl].reset_index(drop=True),
                   cty_d.reset_index(drop=True)], axis=1).astype(float)
    nf = min(5, max(3, len(d)//30))
    try:
        est = DMLEstimator(ml_g="lasso", ml_m="lasso", n_folds=nf, n_reps=3, seed=42)
        res = est.fit(Y=d[y_col].values, D=d[treat_z].values, X=X.values,
                      outcome=y_col, treatment=treat_z, country="POOLED")
        return {"theta":res.theta,"se":res.se,"p_value":res.p_value,
                "ci_lower":res.ci_lower,"ci_upper":res.ci_upper,"n_obs":res.n_obs}
    except Exception as e:
        print(f"      DML error: {e}")
        return None


print("="*65)
print("MS4: REGIME-SPECIFIC DML (orthogonal channel effects)")
print("="*65)
rows = []
for regime_val, regime_label in [(0,"Calm"), (1,"Stress")]:
    sub = panel[panel[STRESS] == regime_val]
    print(f"\n  === {regime_label} regime (n={len(sub)}) ===")
    for treat, label in CHANNELS.items():
        treat_z = ch_z[treat]
        other_z = [ch_z[c] for c in CHANNELS if c != treat]
        res = dml_pooled(sub, treat_z, other_z)
        if res is None:
            print(f"    {label:25s}: skip"); continue
        sig = "***" if res["p_value"]<0.01 else "**" if res["p_value"]<0.05 else "*" if res["p_value"]<0.10 else ""
        print(f"    {label:25s}: theta={res['theta']:+.4f}  SE={res['se']:.4f}  p={res['p_value']:.3f} {sig}")
        rows.append({"regime":regime_label,"channel":treat,"channel_label":label,
                     "theta":round(res["theta"],4),"se":round(res["se"],4),
                     "p_value":round(res["p_value"],4),"signif":sig,
                     "ci_lower":round(res["ci_lower"],4),"ci_upper":round(res["ci_upper"],4),
                     "n_obs":res["n_obs"],"abs_theta":abs(res["theta"])})

df = pd.DataFrame(rows)
df.to_csv(TABLES_DIR / "tableMS4_regime_dml.csv", index=False)
print(f"\nSaved: tableMS4_regime_dml.csv")

# Ranking per regime
print("\n  --- RANKING by |theta| ---")
for regime in ["Calm","Stress"]:
    sub = df[df["regime"]==regime].sort_values("abs_theta", ascending=False)
    print(f"  {regime}: " + " > ".join(
        f"{r['channel_label'].split()[0]}({r['theta']:+.3f}{r['signif']})"
        for _, r in sub.iterrows()))


# ── FIGURE ────────────────────────────────────────────────────────────────────
if not df.empty:
    ch_order = list(CHANNELS.keys())
    labels   = [CHANNELS[c] for c in ch_order]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(ch_order)); h = 0.35

    for regime, off, color in [("Calm", h/2, "#4daf4a"), ("Stress", -h/2, "#d62728")]:
        sub = df[df["regime"]==regime].set_index("channel").reindex(ch_order).reset_index()
        for yi, (_, row) in zip(y+off, sub.iterrows()):
            if pd.isna(row["theta"]): continue
            p = row["p_value"]
            alpha = 0.95 if p<0.05 else 0.6 if p<0.10 else 0.35
            ax.barh(yi, row["theta"], h, color=color, alpha=alpha)
            ax.plot([row["ci_lower"], row["ci_upper"]], [yi, yi], color="k", lw=1)
            if row["signif"]:
                ax.text(row["theta"]+(0.01 if row["theta"]>=0 else -0.01), yi,
                        row["signif"], va="center",
                        ha="left" if row["theta"]>=0 else "right", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4daf4a",label="Calm"),
                       Patch(color="#d62728",label="Stress")], fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("θ  (orthogonal DML effect, std)", fontsize=9)
    ax.set_title("Regime-Specific Orthogonal Channel Effects (DML) — 5-ASEAN\n"
                 "All other channels (incl. persistence) partialled out  |  "
                 "* p<0.10  ** p<0.05  *** p<0.01",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "figMS4_regime_dml.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figMS4_regime_dml.png")

print("\nMS4 HOAN THANH.")
