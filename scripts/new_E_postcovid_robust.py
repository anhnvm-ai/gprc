"""Phase E: Post-COVID Robustness for GPR x PostCOVID effect

Tests:
  E1. Pre vs Post subsample DML (cutoff 2021-12 and 2022-12)
  E2. Two cutoffs: 2021-01 vs 2022-01
  E3. Outlier check: winsorize extreme logRV months, re-run
  E4. Coefficient plot by country × period
  E5. Rolling 24-month GPR effect to detect when it strengthened

Output:
  results/tables/tableE1_subsample_pre_post.csv
  results/tables/tableE2_cutoff_sensitivity.csv
  results/tables/tableE3_outlier_robust.csv
  results/figures/figE1_pre_post_coeff.png
  results/figures/figE2_rolling_gpr_effect.png
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
from scipy.stats import norm
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
CC_COLORS = {"VNM":"#e41a1c","THA":"#377eb8","IDN":"#4daf4a",
             "MYS":"#984ea3","PHL":"#ff7f00"}

# ── Load panel ────────────────────────────────────────────────────────────────
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel = panel[panel["country"].isin(CC5)].copy()

panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_lag1"]  = panel["log_rv"]  # persistence = contemporaneous logRV_t (predicts logRV_{t+1})

# Standardize within country
for col in ["gpr_country","vix_m","dlog_fx_m","log_rv_lag1",
            "log_rv_lead1","sp500_ret_m","us10y_m","wti_ret_m","ret_m"]:
    if col in panel.columns:
        panel[col+"_z"] = panel.groupby("country")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )

OUTCOME   = "log_rv_lead1_z"
TREAT     = "gpr_country_z"
CONTROLS  = ["log_rv_lag1_z","vix_m_z","dlog_fx_m_z",
             "sp500_ret_m_z","us10y_m_z","wti_ret_m_z"]

def run_dml(sub, treat_col, ctrl_cols, label="", n_rep=3):
    ctrl = [c for c in ctrl_cols if c in sub.columns and sub[c].notna().sum()>10]
    df   = sub.dropna(subset=[OUTCOME, treat_col]+ctrl)
    n    = len(df)
    if n < 20:
        return None
    nf = min(5, max(2, n//8))
    try:
        est = DMLEstimator(ml_g="lasso", ml_m="lasso",
                           n_folds=nf, n_reps=n_rep, seed=42)
        res = est.fit(Y=df[OUTCOME].values, D=df[treat_col].values,
                      X=df[ctrl].values,
                      outcome=OUTCOME, treatment=treat_col,
                      country=df["country"].iloc[0] if "country" in df.columns else "?")
        return {"theta":res.theta,"se":res.se,"p_value":res.p_value,
                "ci_lower":res.ci_lower,"ci_upper":res.ci_upper,"n_obs":n}
    except Exception as e:
        print(f"    DML error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# E1. PRE vs POST — two cutoffs (2021-01 and 2022-01)
# ══════════════════════════════════════════════════════════════════════════════
CUTOFFS = {
    "cutoff_2021": ("2021-01-01", "post-COVID = from 2021"),
    "cutoff_2022": ("2022-01-01", "post-COVID = from 2022"),
}
PERIODS = {
    "pre_covid":  ("2010-01-01","2019-12-31"),
    "covid_era":  ("2020-01-01","2021-12-31"),
    "post_covid": ("2022-01-01","2026-04-30"),
}

print("="*65)
print("E1. SUBSAMPLE DML: Pre / COVID / Post per country")
print("="*65)
rows_sub = []
for cc in CC5:
    sub_cc = panel[panel["country"]==cc].copy()
    sub_cc["country"] = cc
    print(f"\n  {COUNTRY_LABEL[cc]}:")
    for period, (t0, t1) in PERIODS.items():
        df = sub_cc.loc[t0:t1]
        res = run_dml(df, TREAT, CONTROLS)
        if res is None:
            print(f"    [{period:12s}]: n<20, skip")
            continue
        sig = "***" if res["p_value"]<0.01 else "**" if res["p_value"]<0.05 else "*" if res["p_value"]<0.10 else ""
        print(f"    [{period:12s}] n={res['n_obs']:3d}: theta={res['theta']:+.4f}  SE={res['se']:.4f}  p={res['p_value']:.3f} {sig}")
        rows_sub.append({"country":cc,"country_label":COUNTRY_LABEL[cc],
                         "period":period,"t_start":t0,"t_end":t1,
                         **{k:round(v,4) for k,v in res.items()},
                         "signif":sig})

df_sub = pd.DataFrame(rows_sub)
df_sub.to_csv(TABLES_DIR/"tableE1_subsample_pre_post.csv", index=False)
print(f"\nSaved: tableE1_subsample_pre_post.csv")


# ══════════════════════════════════════════════════════════════════════════════
# E2. CUTOFF SENSITIVITY — test 2021-01 vs 2022-01 as COVID boundary
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("E2. CUTOFF SENSITIVITY: PostCOVID = 2021 vs 2022")
print("="*65)
rows_cut = []
for cutoff_label, (cutoff_date, desc) in CUTOFFS.items():
    print(f"\n  {desc}:")
    panel_cut = panel.copy()
    panel_cut["post"] = (panel_cut.index >= cutoff_date).astype(float)
    panel_cut["gpr_x_post_z"] = panel_cut[TREAT] * panel_cut["post"]

    for cc in CC5:
        sub = panel_cut[panel_cut["country"]==cc].copy()
        sub["country"] = cc
        ctrl_ext = CONTROLS + ["post"]
        res_base = run_dml(sub, TREAT, ctrl_ext)
        res_int  = run_dml(sub, "gpr_x_post_z", ctrl_ext+[TREAT])
        if res_base and res_int:
            sig_b = "***" if res_base["p_value"]<0.01 else "**" if res_base["p_value"]<0.05 else "*" if res_base["p_value"]<0.10 else ""
            sig_i = "***" if res_int["p_value"]<0.01 else "**" if res_int["p_value"]<0.05 else "*" if res_int["p_value"]<0.10 else ""
            print(f"    {COUNTRY_LABEL[cc]:12s}: baseline={res_base['theta']:+.4f}{sig_b}  x_post={res_int['theta']:+.4f}{sig_i}")
            rows_cut.append({"country":cc,"cutoff":cutoff_label,"cutoff_date":cutoff_date,
                             "theta_base":round(res_base["theta"],4),"p_base":round(res_base["p_value"],4),"sig_base":sig_b,
                             "theta_xpost":round(res_int["theta"],4),"p_xpost":round(res_int["p_value"],4),"sig_xpost":sig_i,
                             "n_obs":res_base["n_obs"]})

df_cut = pd.DataFrame(rows_cut)
df_cut.to_csv(TABLES_DIR/"tableE2_cutoff_sensitivity.csv", index=False)
print(f"\nSaved: tableE2_cutoff_sensitivity.csv")


# ══════════════════════════════════════════════════════════════════════════════
# E3. OUTLIER ROBUSTNESS — winsorize top/bottom 2.5% of logRV
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("E3. OUTLIER ROBUSTNESS: winsorize extreme logRV months")
print("="*65)
rows_out = []
for cc in CC5:
    sub = panel[panel["country"]==cc].copy()
    sub["country"] = cc

    # Winsorize log_rv_lead1 at 2.5% and 97.5%
    lo = sub["log_rv_lead1"].quantile(0.025)
    hi = sub["log_rv_lead1"].quantile(0.975)
    sub["log_rv_lead1_w"] = sub["log_rv_lead1"].clip(lower=lo, upper=hi)
    # Re-standardize
    sub["log_rv_lead1_w_z"] = (sub["log_rv_lead1_w"] - sub["log_rv_lead1_w"].mean()) / \
                               (sub["log_rv_lead1_w"].std() + 1e-12)

    for spec, outcome_col in [("full_sample",   OUTCOME),
                               ("winsorized_2.5", "log_rv_lead1_w_z")]:
        ctrl_use = [c for c in CONTROLS if c in sub.columns and sub[c].notna().sum()>10]
        df_use = sub.dropna(subset=[outcome_col, TREAT]+ctrl_use)
        if len(df_use) < 30: continue
        try:
            nf = min(5, max(2, len(df_use)//12))
            est = DMLEstimator(ml_g="lasso", ml_m="lasso", n_folds=nf, n_reps=3, seed=42)
            res = est.fit(Y=df_use[outcome_col].values, D=df_use[TREAT].values,
                          X=df_use[ctrl_use].values,
                          outcome=outcome_col, treatment=TREAT, country=cc)
            sig = "***" if res.p_value<0.01 else "**" if res.p_value<0.05 else "*" if res.p_value<0.10 else ""
            rows_out.append({"country":cc,"country_label":COUNTRY_LABEL[cc],
                             "spec":spec,"theta":round(res.theta,4),"se":round(res.se,4),
                             "p_value":round(res.p_value,4),"signif":sig,"n_obs":len(df_use)})
        except Exception as e:
            print(f"  {cc} [{spec}]: {e}")

df_out = pd.DataFrame(rows_out)
if not df_out.empty:
    print("\n  Baseline vs Winsorized (GPR_country -> logRV_lead1):")
    pivot = df_out.pivot_table(index="country", columns="spec",
                               values=["theta","p_value"], aggfunc="first")
    print(pivot.round(4).to_string())
    df_out.to_csv(TABLES_DIR/"tableE3_outlier_robust.csv", index=False)
    print(f"\nSaved: tableE3_outlier_robust.csv")


# ══════════════════════════════════════════════════════════════════════════════
# E4. FIGURE: Pre/Post coefficient plot per country
# ══════════════════════════════════════════════════════════════════════════════
if not df_sub.empty:
    period_order  = ["pre_covid","covid_era","post_covid"]
    period_labels = {"pre_covid":"Pre-COVID\n(2010–19)",
                     "covid_era":"COVID\n(2020–21)",
                     "post_covid":"Post-COVID\n(2022–26)"}

    fig, axes = plt.subplots(1, len(CC5), figsize=(len(CC5)*3.0, 5.5), sharey=False)
    fig.suptitle(
        "GPR Country → logRV_{t+1}: Pre vs Post-COVID by country\n"
        "DML-Lasso | 95% CI | Red=p<0.05  Orange=p<0.10  Blue=n.s.",
        fontsize=10, fontweight="bold"
    )

    for ax, cc in zip(axes, CC5):
        sub = df_sub[df_sub["country"]==cc].set_index("period").reindex(period_order)
        sub = sub.reset_index().rename(columns={"index":"period"})

        for i, row in sub.iterrows():
            if pd.isna(row.get("theta")): continue
            p   = row.get("p_value", 1.0)
            col = "#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0"
            ci_lo = row.get("ci_lower", row["theta"] - 1.96*row["se"])
            ci_hi = row.get("ci_upper", row["theta"] + 1.96*row["se"])
            ax.errorbar(row["theta"], i,
                        xerr=[[row["theta"]-ci_lo],[ci_hi-row["theta"]]],
                        fmt="o", color=col, capsize=5, lw=2.5, ms=8)
            sig = row.get("signif","")
            if sig:
                ax.text(ci_hi+0.02, i, sig, va="center", fontsize=10, color=col)

        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.set_yticks(range(len(period_order)))
        ax.set_yticklabels([period_labels[p] for p in period_order], fontsize=8)
        ax.set_title(COUNTRY_LABEL[cc], fontsize=9, fontweight="bold",
                     color=CC_COLORS[cc])
        ax.set_xlabel("θ (std)", fontsize=8)
        ax.grid(axis="x", alpha=0.25)
        # shade post-COVID row
        ax.axhspan(1.5, 2.5, color="#FFF3E0", alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(FIGS_DIR/"figE1_pre_post_coeff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figE1_pre_post_coeff.png")


# ══════════════════════════════════════════════════════════════════════════════
# E5. ROLLING 24-MONTH GPR EFFECT (OLS, for visualization)
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("E5. ROLLING 24-MONTH GPR EFFECT (OLS beta over time)")
print("="*65)

import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(12, 5))
WINDOW = 30   # months

for cc in CC5:
    sub = panel[panel["country"]==cc].dropna(
        subset=[OUTCOME, TREAT]+CONTROLS).copy()
    sub = sub.sort_index()
    dates, betas, pvals = [], [], []

    for end_idx in range(WINDOW, len(sub)):
        window = sub.iloc[end_idx-WINDOW:end_idx]
        Y_w = window[OUTCOME].values
        X_w = sm.add_constant(window[[TREAT]+CONTROLS].values)
        try:
            m = sm.OLS(Y_w, X_w).fit()
            betas.append(m.params[1])
            pvals.append(m.pvalues[1])
            dates.append(sub.index[end_idx])
        except:
            pass

    if not dates:
        continue
    betas = np.array(betas)
    pvals = np.array(pvals)
    sig_mask = np.array(pvals) < 0.10

    ax.plot(dates, betas, color=CC_COLORS[cc], lw=1.5, alpha=0.8,
            label=COUNTRY_LABEL[cc])
    # mark significant periods
    ax.scatter([d for d,s in zip(dates,sig_mask) if s],
               [b for b,s in zip(betas,sig_mask) if s],
               color=CC_COLORS[cc], s=18, zorder=5, alpha=0.7)

ax.axhline(0, color="k", ls="--", lw=0.8)
ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
           color="#BBDEFB", alpha=0.3, label="COVID period")
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2026-12-31"),
           color="#FFF9C4", alpha=0.3, label="Post-COVID")
ax.set_ylabel("Rolling 30-month OLS β (GPR_country → logRV_t+1, std)", fontsize=9)
ax.set_title("Rolling GPR Effect Over Time — 5-ASEAN\n"
             "Dots = significant at p<0.10 | Window = 30 months",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, ncol=4)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIGS_DIR/"figE2_rolling_gpr_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figE2_rolling_gpr_effect.png")


# ── Narrative summary ─────────────────────────────────────────────────────────
print("\n"+"="*65)
print("NARRATIVE: GPR x PostCOVID robustness summary")
print("="*65)
if not df_sub.empty:
    print("\n  Pre-COVID theta (GPR_country -> logRV_lead1):")
    pre = df_sub[df_sub["period"]=="pre_covid"][["country","theta","p_value","signif","n_obs"]]
    print(pre.to_string(index=False))
    print("\n  Post-COVID theta:")
    post = df_sub[df_sub["period"]=="post_covid"][["country","theta","p_value","signif","n_obs"]]
    print(post.to_string(index=False))

if not df_cut.empty:
    print("\n  GPR x PostCOVID interaction — 2022 cutoff:")
    c22 = df_cut[df_cut["cutoff"]=="cutoff_2022"][["country","theta_xpost","p_xpost","sig_xpost"]]
    print(c22.to_string(index=False))

print("\nPhase E HOAN THANH.")
