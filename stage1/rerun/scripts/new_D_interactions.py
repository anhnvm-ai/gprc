"""Phase D: Interactions — GPR×FX, GPR×VIX, Post-COVID amplification

Tests:
  D1. GPR × dlogFX   : does GPR amplify FX shock impact on volatility?
  D2. GPR × VIX      : does GPR amplify global risk aversion?
  D3. GPR × PostCOVID: did GPR effect strengthen after COVID?

Method: DML with interaction term as treatment
  - D_interact = GPR_country × Channel
  - Controls   = GPR_country, Channel, other channels, macro

Output:
  results/tables/tableD_interactions.csv
  results/figures/figD_interactions.png
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

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_monthly.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CC5 = ["VNM","THA","IDN","MYS","PHL"]
COUNTRY_LABEL = {"VNM":"Vietnam","THA":"Thailand","IDN":"Indonesia",
                 "MYS":"Malaysia","PHL":"Philippines"}

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel = panel[panel["country"].isin(CC5)].copy()

panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_lag1"]  = panel["log_rv"]  # persistence = contemporaneous logRV_t (predicts logRV_{t+1})
panel["post_covid"]   = (panel.index >= "2022-01-01").astype(float)

# Standardize channels within country
for col in ["gpr_country","vix_m","dlog_fx_m","log_rv_lag1","log_rv_lead1",
            "sp500_ret_m","us10y_m","wti_ret_m","ret_m"]:
    if col in panel.columns:
        panel[col+"_z"] = panel.groupby("country")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )

# Interaction terms (unstandardized, then let DML handle them)
panel["gpr_x_fx"]       = panel["gpr_country_z"] * panel["dlog_fx_m_z"]
panel["gpr_x_vix"]      = panel["gpr_country_z"] * panel["vix_m_z"]
panel["gpr_x_postcovid"]= panel["gpr_country_z"] * panel["post_covid"]

OUTCOME  = "log_rv_lead1_z"
CONTROLS = ["log_rv_lag1_z","sp500_ret_m_z","us10y_m_z","wti_ret_m_z","ret_m_z"]

INTERACT_SPECS = {
    "GPR x FX":        ("gpr_x_fx",        ["gpr_country_z","dlog_fx_m_z","vix_m_z"]),
    "GPR x VIX":       ("gpr_x_vix",       ["gpr_country_z","vix_m_z","dlog_fx_m_z"]),
    "GPR x PostCOVID": ("gpr_x_postcovid", ["gpr_country_z","post_covid","vix_m_z","dlog_fx_m_z"]),
}

rows = []
for cc in CC5:
    sub = panel[panel["country"]==cc].copy()
    print(f"\n{'='*55}")
    print(f"{COUNTRY_LABEL[cc]}")
    print("="*55)

    for label, (treat_col, extra_ctrl) in INTERACT_SPECS.items():
        ctrl = [c for c in CONTROLS + extra_ctrl
                if c in sub.columns and sub[c].notna().sum() > 20]
        df = sub.dropna(subset=[OUTCOME, treat_col] + ctrl)
        n  = len(df)
        if n < 30:
            print(f"  {label}: n={n} skip")
            continue
        try:
            n_folds = min(5, max(2, n//12))
            est = DMLEstimator(ml_g="lasso", ml_m="lasso",
                               n_folds=n_folds, n_reps=3, seed=42)
            res = est.fit(Y=df[OUTCOME].values, D=df[treat_col].values,
                          X=df[ctrl].values,
                          outcome=OUTCOME, treatment=treat_col, country=cc)
            sig = "***" if res.p_value<0.01 else "**" if res.p_value<0.05 else "*" if res.p_value<0.10 else ""
            print(f"  {label:20s}: theta={res.theta:+.4f}  SE={res.se:.4f}  p={res.p_value:.3f} {sig}  n={n}")
            rows.append({"country":cc,"country_label":COUNTRY_LABEL[cc],
                         "interaction":label,"treat_col":treat_col,
                         "theta":round(res.theta,4),"se":round(res.se,4),
                         "p_value":round(res.p_value,4),"signif":sig,"n_obs":n})
        except Exception as e:
            print(f"  {label}: ERROR {e}")

df_out = pd.DataFrame(rows)
if not df_out.empty:
    df_out.to_csv(TABLES_DIR / "tableD_interactions.csv", index=False)
    print(f"\nSaved: tableD_interactions.csv")


# ── FIGURE: interaction effects per country ───────────────────────────────────
if not df_out.empty:
    intx_list = list(INTERACT_SPECS.keys())
    colors = {"VNM":"#e41a1c","THA":"#377eb8","IDN":"#4daf4a",
               "MYS":"#984ea3","PHL":"#ff7f00"}

    fig, axes = plt.subplots(1, len(intx_list), figsize=(len(intx_list)*4, 5))
    if len(intx_list) == 1:
        axes = [axes]
    fig.suptitle("Interaction Effects: GPR Amplification\n"
                 "θ > 0 = GPR amplifies channel  |  DML-Lasso  |  95% CI",
                 fontsize=10, fontweight="bold")

    for ax, intx in zip(axes, intx_list):
        sub = df_out[df_out["interaction"]==intx].set_index("country").reindex(CC5).reset_index()
        sub = sub.dropna(subset=["theta"])

        for i, row in sub.iterrows():
            p   = row.get("p_value",1.0)
            col = "#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0"
            ax.errorbar(row["theta"], i, xerr=1.96*row["se"],
                        fmt="o", color=col, capsize=4, lw=2, ms=7)
            if row.get("signif",""):
                ax.text(row["theta"]+1.96*row["se"]+0.01, i,
                        row["signif"], va="center", fontsize=9, color=col)

        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([COUNTRY_LABEL.get(c,c) for c in sub["country"]], fontsize=9)
        ax.set_title(intx, fontsize=9, fontweight="bold")
        ax.set_xlabel("θ (std)", fontsize=8)
        ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "figD_interactions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figD_interactions.png")

print("\nPhase D HOAN THANH.")
