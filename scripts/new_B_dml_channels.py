"""Phase B: DML — Orthogonal causal effect per transmission channel

For each channel D in {logRV_lag, GPR_country, VIX, dlogFX}:
  - Outcome Y = logRV_{t+1}
  - Controls X = other 3 channels + macro controls + country dummies
  - DML gives θ_D: causal effect of D on Y AFTER partialling out all other channels

This answers: "Does GPR country add causal power BEYOND volatility persistence and VIX?"

Output:
  results/tables/tableB1_dml_channels_pooled.csv
  results/tables/tableB2_dml_channels_by_country.csv
  results/figures/figB_dml_channel_comparison.png
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
from statsmodels.stats.multitest import multipletests
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

# ── Load panel ────────────────────────────────────────────────────────────────
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel = panel[panel["country"].isin(CC5)].copy()

panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_lag1"]  = panel["log_rv"]  # persistence = contemporaneous logRV_t (predicts logRV_{t+1})

CHANNELS = {
    "log_rv_lag1":  "Volatility Persistence",
    "gpr_country":  "GPR Country",
    "vix_m":        "VIX (Global Risk)",
    "dlog_fx_m":    "FX Shock",
}
MACRO_CONTROLS = ["sp500_ret_m","us10y_m","wti_ret_m","ret_m"]
OUTCOME = "log_rv_lead1"

LEARNERS = ["lasso","ridge","rf"]


# ── Standardize within country ────────────────────────────────────────────────
for col in list(CHANNELS.keys()) + MACRO_CONTROLS + [OUTCOME]:
    if col in panel.columns:
        panel[col+"_z"] = panel.groupby("country")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )

ch_z     = {k: k+"_z" for k in CHANNELS}
macro_z  = [c+"_z" for c in MACRO_CONTROLS if c in panel.columns]
outcome_z = OUTCOME + "_z"


# ── DML runner ────────────────────────────────────────────────────────────────
def run_dml_channel(sub, treat_col_z, label, other_ch_z, ml="lasso"):
    all_controls = other_ch_z + macro_z
    ctrl = [c for c in all_controls if c in sub.columns and sub[c].notna().sum() > 20]
    df = sub.dropna(subset=[outcome_z, treat_col_z] + ctrl)
    n = len(df)
    if n < 30:
        return None
    n_folds = min(5, max(2, n//12))
    est = DMLEstimator(ml_g=ml, ml_m=ml, n_folds=n_folds, n_reps=3, seed=42)
    try:
        res = est.fit(Y=df[outcome_z].values, D=df[treat_col_z].values,
                      X=df[ctrl].values,
                      outcome=outcome_z, treatment=treat_col_z,
                      country=sub["country"].iloc[0] if "country" in sub.columns else "POOLED")
        return {"theta": res.theta, "se": res.se, "p_value": res.p_value,
                "n_obs": res.n_obs, "ml": ml, "ci_lower": res.ci_lower, "ci_upper": res.ci_upper}
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# A. POOLED — all 5 countries together with country dummies
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("PHASE B — DML CHANNEL COMPARISON (standardized variables)")
print("="*65)
print("\n--- POOLED (5-ASEAN + country dummies) ---")

# Add country dummies to controls
cty_dummies = pd.get_dummies(panel["country"], prefix="cty", drop_first=True)
cty_cols    = list(cty_dummies.columns)
panel_pool  = pd.concat([panel.reset_index(), cty_dummies.reset_index(drop=True)], axis=1)
panel_pool  = panel_pool.set_index("date")

rows_pool = []
for treat_col, label in CHANNELS.items():
    treat_z = treat_col + "_z"
    other_z = [ch_z[c] for c in CHANNELS if c != treat_col]
    ctrl_pool = other_z + macro_z + cty_cols

    ctrl_pool = [c for c in ctrl_pool if c in panel_pool.columns
                 and panel_pool[c].notna().sum() > 100]
    df = panel_pool.dropna(subset=[outcome_z, treat_z] + ctrl_pool)
    if len(df) < 50:
        continue

    for ml in LEARNERS:
        est = DMLEstimator(ml_g=ml, ml_m=ml, n_folds=5, n_reps=3, seed=42)
        try:
            res = est.fit(Y=df[outcome_z].values, D=df[treat_z].values,
                          X=df[ctrl_pool].values,
                          outcome=outcome_z, treatment=treat_z, country="POOLED")
            sig = "***" if res.p_value<0.01 else "**" if res.p_value<0.05 else "*" if res.p_value<0.10 else ""
            print(f"  {label:25s} [{ml:5s}]: theta={res.theta:+.4f}  SE={res.se:.4f}  p={res.p_value:.3f} {sig}")
            rows_pool.append({"country":"POOLED","channel":treat_col,"channel_label":label,
                              "ml":ml,"theta":res.theta,"se":res.se,
                              "p_value":res.p_value,"signif":sig,"n_obs":res.n_obs})
        except Exception as e:
            print(f"  {label} [{ml}]: {e}")

df_pool = pd.DataFrame(rows_pool)


# ══════════════════════════════════════════════════════════════════════════════
# B. PER COUNTRY
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- PER COUNTRY (Lasso + Ridge) ---")
rows_cc = []
for cc in CC5:
    sub = panel[panel["country"] == cc].copy()
    sub_reset = sub.reset_index()
    sub_reset["country"] = cc
    print(f"\n  {COUNTRY_LABEL[cc]} (n={len(sub.dropna(subset=[outcome_z]))}):")

    for treat_col, label in CHANNELS.items():
        treat_z = treat_col + "_z"
        other_z = [ch_z[c] for c in CHANNELS if c != treat_col]

        for ml in ["lasso","ridge"]:
            res = run_dml_channel(sub_reset, treat_z, label, other_z, ml=ml)
            if res is None:
                continue
            sig = "***" if res["p_value"]<0.01 else "**" if res["p_value"]<0.05 else "*" if res["p_value"]<0.10 else ""
            if ml == "lasso":
                print(f"    {label:25s}: theta={res['theta']:+.4f}  p={res['p_value']:.3f} {sig}  [n={res['n_obs']}]")
            rows_cc.append({"country":cc, "country_label":COUNTRY_LABEL[cc],
                            "channel":treat_col, "channel_label":label,
                            "ml":ml, "theta":round(res["theta"],4), "se":round(res["se"],4),
                            "p_value":round(res["p_value"],4), "signif":sig,
                            "ci_lower":round(res["ci_lower"],4), "ci_upper":round(res["ci_upper"],4),
                            "n_obs":res["n_obs"]})

df_cc = pd.DataFrame(rows_cc)

# BH correction per channel
if not df_cc.empty:
    rows_corr = []
    for ch in df_cc["channel"].unique():
        sub_ch = df_cc[(df_cc["channel"]==ch) & (df_cc["ml"]=="lasso")].copy()
        pv = sub_ch["p_value"].values
        if len(pv) > 1:
            _, p_adj, _, _ = multipletests(pv, alpha=0.05, method="fdr_bh")
            sub_ch["p_bh"] = p_adj
        else:
            sub_ch["p_bh"] = sub_ch["p_value"]
        rows_corr.append(sub_ch)
    df_cc_bh = pd.concat(rows_corr).reset_index(drop=True)
else:
    df_cc_bh = df_cc


# ── Summary: average |θ| per channel ─────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY: Average |theta| per channel (Lasso, across countries)")
print("="*65)
if not df_cc.empty:
    summ = df_cc[df_cc["ml"]=="lasso"].groupby("channel").agg(
        avg_abs_theta=("theta", lambda x: np.mean(np.abs(x))),
        pct_sig=("signif", lambda x: (x!="").mean()*100),
        channel_label=("channel_label","first")
    ).sort_values("avg_abs_theta", ascending=False)
    print(summ[["channel_label","avg_abs_theta","pct_sig"]].round(4).to_string())


# ── Save ─────────────────────────────────────────────────────────────────────
df_pool.to_csv(TABLES_DIR / "tableB1_dml_channels_pooled.csv", index=False)
df_cc_bh.to_csv(TABLES_DIR / "tableB2_dml_channels_by_country.csv", index=False)
print(f"\nSaved: tableB1_dml_channels_pooled.csv")
print(f"Saved: tableB2_dml_channels_by_country.csv")


# ── FIGURE ────────────────────────────────────────────────────────────────────
if not df_cc_bh.empty:
    lasso_cc = df_cc_bh[df_cc_bh["ml"]=="lasso"]
    ch_list  = list(CHANNELS.keys())
    n_ch = len(ch_list)

    SHORT = {"log_rv_lag1": "Volatility persistence", "gpr_country": "GPRC",
             "vix_m": "VIX", "dlog_fx_m": "FX shock"}
    fig, axes = plt.subplots(1, n_ch, figsize=(n_ch * 3.9, 5.6), sharey=True)

    for ax, ch in zip(axes, ch_list):
        label = SHORT.get(ch, CHANNELS[ch])
        sub   = lasso_cc[lasso_cc["channel"]==ch].set_index("country").reindex(CC5).reset_index()
        sub   = sub.dropna(subset=["theta"])

        for i, row in sub.iterrows():
            p = row.get("p_value", 1.0)
            col = "#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0"
            ci_lo = row.get("ci_lower", row["theta"]-1.96*row["se"])
            ci_hi = row.get("ci_upper", row["theta"]+1.96*row["se"])
            ax.errorbar(row["theta"], i,
                        xerr=[[row["theta"]-ci_lo],[ci_hi-row["theta"]]],
                        fmt="o", color=col, capsize=5, lw=2.2, ms=8)
            ax.text(ci_hi + 0.012, i, row.get("signif",""), va="center",
                    fontsize=13, color=col)

        ax.axvline(0, color="k", ls="--", lw=1.0, alpha=0.5)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([COUNTRY_LABEL.get(c,c) for c in sub["country"]])
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel(r"DML estimate $\hat\theta$")
        ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "figB_dml_channel_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGS_DIR / "figB_dml_channel_comparison.pdf", bbox_inches="tight")  # vector PDF
    plt.close()
    print(f"Saved: figB_dml_channel_comparison.png")

print("\nPhase B HOAN THANH.")
