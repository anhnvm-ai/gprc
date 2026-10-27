"""Phase G — Post-COVID Robustness for GPR Effect

1. Pre-COVID (2010-2019) vs Post-COVID (2022-2026) DML per country
2. Test two cutoffs: 2021-01 and 2022-01
3. Check extreme months (COVID spike 2020-03): drop and re-run
4. Per-country coefficient plot Pre vs Post

Output:
  results/tables/tableG1_gpr_pre_post.csv
  results/tables/tableG2_cutoff_sensitivity.csv
  results/figures/figG_gpr_regime.png
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

for col in ["log_rv_lag1","gpr_country","vix_m","dlog_fx_m","log_rv_lead1",
            "sp500_ret_m","us10y_m","wti_ret_m","ret_m"]:
    if col in panel.columns:
        panel[col+"_z"] = panel.groupby("country")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )

OUTCOME  = "log_rv_lead1_z"
TREAT    = "gpr_country_z"
CONTROLS = ["log_rv_lag1_z","vix_m_z","dlog_fx_m_z",
            "sp500_ret_m_z","us10y_m_z","wti_ret_m_z","ret_m_z"]

def run_dml(sub, label):
    ctrl = [c for c in CONTROLS if c in sub.columns and sub[c].notna().sum() > 20]
    df = sub.dropna(subset=[OUTCOME, TREAT] + ctrl)
    n = len(df)
    if n < 20:
        return None
    n_folds = min(5, max(2, n//8))
    try:
        est = DMLEstimator(ml_g="lasso", ml_m="lasso",
                           n_folds=n_folds, n_reps=3, seed=42)
        res = est.fit(Y=df[OUTCOME].values, D=df[TREAT].values,
                      X=df[ctrl].values, outcome=OUTCOME,
                      treatment=TREAT, country=label)
        sig = "***" if res.p_value<0.01 else "**" if res.p_value<0.05 else "*" if res.p_value<0.10 else ""
        return {"theta": round(res.theta,4), "se": round(res.se,4),
                "p_value": round(res.p_value,4), "signif": sig, "n": n}
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# G1. Pre-COVID vs Post-COVID per country (cutoff 2022-01)
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("G1. PRE-COVID (2010-2019) vs POST-COVID (2022-2026)")
print("="*65)
rows_pp = []
for cc in CC5:
    sub_cc = panel[panel["country"]==cc]
    pre  = sub_cc["2010-01-01":"2019-12-31"]
    post = sub_cc["2022-01-01":"2026-04-30"]
    full = sub_cc

    print(f"\n  {COUNTRY_LABEL[cc]}:")
    for period, data in [("Pre-COVID  ", pre), ("Post-COVID ", post), ("Full sample", full)]:
        res = run_dml(data.copy(), f"{cc}_{period.strip()}")
        if res:
            print(f"    {period}: theta={res['theta']:+.4f}  SE={res['se']:.4f}  "
                  f"p={res['p_value']:.3f}{res['signif']:3s}  n={res['n']}")
            rows_pp.append({"country":cc, "country_label":COUNTRY_LABEL[cc],
                            "period":period.strip(), **res})
        else:
            print(f"    {period}: insufficient data")

df_pp = pd.DataFrame(rows_pp)


# ══════════════════════════════════════════════════════════════════════════════
# G2. Cutoff sensitivity: 2021-01 vs 2022-01 vs 2023-01
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("G2. CUTOFF SENSITIVITY (Post-COVID period)")
print("="*65)
rows_cut = []
cutoffs = {"Post_2021": "2021-01-01", "Post_2022": "2022-01-01", "Post_2023": "2023-01-01"}
for cc in CC5:
    sub_cc = panel[panel["country"]==cc]
    print(f"\n  {COUNTRY_LABEL[cc]}:")
    for cut_label, cut_date in cutoffs.items():
        sub_post = sub_cc[cut_date:]
        res = run_dml(sub_post.copy(), f"{cc}_{cut_label}")
        if res:
            print(f"    {cut_label} (from {cut_date[:7]}): theta={res['theta']:+.4f}  "
                  f"p={res['p_value']:.3f}{res['signif']:3s}  n={res['n']}")
            rows_cut.append({"country":cc, "cutoff":cut_label,
                             "cutoff_date":cut_date, **res})

df_cut = pd.DataFrame(rows_cut)


# ══════════════════════════════════════════════════════════════════════════════
# G3. Extreme month robustness: drop 2020-03 (COVID crash)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("G3. DROP EXTREME MONTHS: 2020-03 (COVID crash)")
print("="*65)
rows_ext = []
extreme_months = ["2020-03", "2020-04", "2022-02"]   # COVID peak + Ukraine
for cc in CC5:
    sub_cc = panel[panel["country"]==cc]
    # Drop extreme months
    mask = ~sub_cc.index.strftime("%Y-%m").isin(extreme_months)
    sub_drop = sub_cc[mask]

    res_full = run_dml(sub_cc.copy(), cc)
    res_drop = run_dml(sub_drop.copy(), f"{cc}_drop_extreme")

    if res_full and res_drop:
        diff = res_drop["theta"] - res_full["theta"]
        print(f"  {COUNTRY_LABEL[cc]:12s}: Full theta={res_full['theta']:+.4f}{res_full['signif']:3s}  "
              f"Drop extreme theta={res_drop['theta']:+.4f}{res_drop['signif']:3s}  "
              f"Diff={diff:+.4f}")
        rows_ext.append({"country":cc, "spec":"full_sample", **res_full})
        rows_ext.append({"country":cc, "spec":"drop_extreme_months", **res_drop})

df_ext = pd.DataFrame(rows_ext)


# ── Save ─────────────────────────────────────────────────────────────────────
df_pp.to_csv(TABLES_DIR / "tableG1_gpr_pre_post.csv", index=False)
df_cut.to_csv(TABLES_DIR / "tableG2_cutoff_sensitivity.csv", index=False)
df_ext.to_csv(TABLES_DIR / "tableG3_extreme_months.csv", index=False)
print(f"\nSaved: tableG1_gpr_pre_post.csv")
print(f"Saved: tableG2_cutoff_sensitivity.csv")
print(f"Saved: tableG3_extreme_months.csv")


# ── FIGURE: Pre vs Post coefficient plot ─────────────────────────────────────
if not df_pp.empty:
    pre_data  = df_pp[df_pp["period"]=="Pre-COVID"].set_index("country").reindex(CC5)
    post_data = df_pp[df_pp["period"]=="Post-COVID"].set_index("country").reindex(CC5)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CC5))
    width = 0.35

    def plot_bars(data, offset, label, color):
        vals = data["theta"].values.astype(float)
        ses  = data["se"].values.astype(float)
        pvals = data["p_value"].values.astype(float) if "p_value" in data.columns else [1]*len(vals)
        bars = ax.bar(x+offset, vals, width, yerr=1.96*ses,
                      color=color, alpha=0.8, capsize=4,
                      error_kw={"lw":1.2}, label=label)
        for xi, (v, p) in enumerate(zip(vals, pvals)):
            if not np.isnan(v) and not np.isnan(p) and p < 0.10:
                marker = "★" if p < 0.05 else "·"
                ax.text(xi+offset, v + (1.96*ses[xi] if not np.isnan(ses[xi]) else 0) + 0.005,
                        marker, ha="center", fontsize=10, color="red")
        return bars

    plot_bars(pre_data,  -width/2, "Pre-COVID (2010–2019)", "#1976D2")
    plot_bars(post_data,  width/2, "Post-COVID (2022–2026)", "#D32F2F")

    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_LABEL[c] for c in CC5], fontsize=10)
    ax.set_ylabel("DML θ (GPR_country → logRV_{t+1}, std)", fontsize=9)
    ax.set_title("GPR Country Effect: Pre-COVID vs Post-COVID\n"
                 "★ = p<0.05  ·= p<0.10  |  Error bars = 95% CI  |  DML-Lasso",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "figG_gpr_regime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figG_gpr_regime.png")

# Cutoff sensitivity plot
if not df_cut.empty:
    fig, axes = plt.subplots(1, len(CC5), figsize=(len(CC5)*2.8, 4.5), sharey=False)
    cut_colors = {"Post_2021":"#FF9800","Post_2022":"#D32F2F","Post_2023":"#7B1FA2"}
    for ax, cc in zip(axes, CC5):
        sub = df_cut[df_cut["country"]==cc]
        for i, (_, row) in enumerate(sub.iterrows()):
            col = cut_colors.get(row["cutoff"],"grey")
            ax.errorbar(row["theta"], i, xerr=1.96*row["se"],
                        fmt="o", color=col, capsize=4, ms=7, lw=2)
            ax.text(row["theta"]+1.96*row["se"]+0.01, i,
                    row.get("signif",""), va="center", fontsize=9, color=col)
        ax.axvline(0, color="k", ls="--", lw=0.7)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([r["cutoff"] for _, r in sub.iterrows()], fontsize=7)
        ax.set_title(COUNTRY_LABEL[cc], fontsize=8, fontweight="bold")
        ax.set_xlabel("θ", fontsize=8)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("GPR Country Effect: Cutoff Sensitivity (Post-COVID start)\n"
                 "Results should be stable across 2021/2022/2023 cutoffs",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "figG2_cutoff_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figG2_cutoff_sensitivity.png")

print("\nPhase G HOAN THANH.")
