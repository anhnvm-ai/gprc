"""MS3: GPRC x Stress Interaction + Persistence Contamination Test (Step 3 & H5)

The central identification question of the upgraded paper:
  Is the GPR effect REAL, or is it "persistence contamination"?

Interaction model:
  logRV_{i,t+1} = b0 + b1*GPRC + b2*Stress + b3*(GPRC x Stress)
                  + b4*logRV_t + controls + a_i + u

Focus: b3 (does GPR strengthen in stress?)
H5 test: compare b1, b3 WITH vs WITHOUT logRV_t control.
  - If GPRC x Stress survives adding logRV_t -> REAL regime-dependent GPR channel
  - If it collapses -> apparent GPR amplification was persistence contamination

Estimators: OLS (clustered) + DML (orthogonal, treatment = GPRC x Stress)

Outputs:
  results/tables/tableMS3_interaction.csv
  results/tables/tableMS3_contamination.csv
  results/figures/figMS3_contamination.png
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
from src.models.dml import DMLEstimator

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
panel["log_rv_t"]     = panel["log_rv"]

STRESS = "stress_common"

# Standardize within country
for col in ["gpr_country","vix_m","dlog_fx_m","log_rv_t","log_rv_lead1",
            "sp500_ret_m","us10y_m","wti_ret_m","ret_m"]:
    panel[col+"_z"] = panel.groupby("country")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))

panel["gpr_x_stress"] = panel["gpr_country_z"] * panel[STRESS]

OUTCOME = "log_rv_lead1_z"
BASE_CTRL = ["vix_m_z","dlog_fx_m_z","sp500_ret_m_z","us10y_m_z","wti_ret_m_z","ret_m_z"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. OLS interaction: WITHOUT vs WITH persistence control
# ══════════════════════════════════════════════════════════════════════════════
def run_ols_interaction(with_persistence: bool):
    cols = ["gpr_country_z", STRESS, "gpr_x_stress"] + BASE_CTRL
    if with_persistence:
        cols = ["gpr_country_z", STRESS, "gpr_x_stress", "log_rv_t_z"] + BASE_CTRL
    df = panel.dropna(subset=[OUTCOME]+cols).copy()
    cty_d = pd.get_dummies(df["country"], prefix="cty", drop_first=True).astype(float)
    X = pd.concat([df[cols].reset_index(drop=True),
                   cty_d.reset_index(drop=True)], axis=1).astype(float)
    Xc = sm.add_constant(X.values)
    m = OLS(df[OUTCOME].values, Xc).fit(cov_type="cluster",
                                        cov_kwds={"groups": df["country"].values})
    names = ["const"] + cols + list(cty_d.columns)
    out = {}
    for key in ["gpr_country_z", STRESS, "gpr_x_stress", "log_rv_t_z"]:
        if key in names:
            i = names.index(key)
            out[key] = (m.params[i], m.bse[i], m.pvalues[i])
    return out, len(df), m.rsquared

print("="*65)
print("MS3-1. OLS INTERACTION: Persistence Contamination Test")
print("="*65)
rows_int = []
for wp, lab in [(False,"WITHOUT logRV_t"), (True,"WITH logRV_t")]:
    out, n, r2 = run_ols_interaction(wp)
    print(f"\n  --- {lab} (n={n}, R2={r2:.3f}) ---")
    for key in ["gpr_country_z", STRESS, "gpr_x_stress", "log_rv_t_z"]:
        if key in out:
            b, se, p = out[key]
            sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
            name = {"gpr_country_z":"GPRC (b1)", STRESS:"Stress (b2)",
                    "gpr_x_stress":"GPRC x Stress (b3)","log_rv_t_z":"logRV_t (b4)"}[key]
            print(f"    {name:22s}: beta={b:+.4f}  SE={se:.4f}  p={p:.3f} {sig}")
            rows_int.append({"spec":lab,"term":name,"beta":round(b,4),
                             "se":round(se,4),"p_value":round(p,4),"signif":sig,"n":n})

df_int = pd.DataFrame(rows_int)
df_int.to_csv(TABLES_DIR / "tableMS3_interaction.csv", index=False)
print(f"\nSaved: tableMS3_interaction.csv")

# Contamination verdict
b3_without = df_int[(df_int["spec"]=="WITHOUT logRV_t") & (df_int["term"]=="GPRC x Stress (b3)")]
b3_with    = df_int[(df_int["spec"]=="WITH logRV_t")    & (df_int["term"]=="GPRC x Stress (b3)")]
if len(b3_without) and len(b3_with):
    bw, pw = float(b3_without["beta"].iloc[0]), float(b3_without["p_value"].iloc[0])
    bc, pc = float(b3_with["beta"].iloc[0]),    float(b3_with["p_value"].iloc[0])
    shrink = (1 - abs(bc)/abs(bw))*100 if bw != 0 else np.nan
    print("\n  >>> CONTAMINATION TEST (b3 = GPRC x Stress):")
    print(f"      WITHOUT persistence: b3={bw:+.4f} (p={pw:.3f})")
    print(f"      WITH    persistence: b3={bc:+.4f} (p={pc:.3f})")
    print(f"      Shrinkage: {shrink:.1f}%")
    if pc < 0.10 and abs(bc) > 0.5*abs(bw):
        verdict = "REAL regime-dependent GPR channel (survives persistence control)"
    elif pc >= 0.10 and pw < 0.10:
        verdict = "PERSISTENCE CONTAMINATION (GPR x Stress collapses after control)"
    else:
        verdict = "PARTIAL / weak"
    print(f"      VERDICT: {verdict}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DML orthogonal: treatment = GPRC x Stress, controls include logRV_t
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("MS3-2. DML ORTHOGONAL: GPRC x Stress (controlling persistence)")
print("="*65)
rows_dml = []
for treat, ctrl_extra, lab in [
    ("gpr_x_stress", ["gpr_country_z", STRESS],                "GPRC x Stress (no persist ctrl)"),
    ("gpr_x_stress", ["gpr_country_z", STRESS, "log_rv_t_z"],  "GPRC x Stress (+ persist ctrl)"),
    ("gpr_country_z",[STRESS, "log_rv_t_z"],                    "GPRC main (+ persist ctrl)"),
]:
    ctrl = ctrl_extra + BASE_CTRL
    df = panel.dropna(subset=[OUTCOME, treat]+ctrl).copy()
    # pooled with country dummies
    cty_d = pd.get_dummies(df["country"], prefix="cty", drop_first=True).astype(float)
    Xmat = pd.concat([df[ctrl].reset_index(drop=True),
                      cty_d.reset_index(drop=True)], axis=1).astype(float)
    try:
        est = DMLEstimator(ml_g="lasso", ml_m="lasso", n_folds=5, n_reps=3, seed=42)
        res = est.fit(Y=df[OUTCOME].values, D=df[treat].values, X=Xmat.values,
                      outcome=OUTCOME, treatment=treat, country="POOLED")
        sig = "***" if res.p_value<0.01 else "**" if res.p_value<0.05 else "*" if res.p_value<0.10 else ""
        print(f"  {lab:34s}: theta={res.theta:+.4f}  SE={res.se:.4f}  p={res.p_value:.3f} {sig}  n={res.n_obs}")
        rows_dml.append({"spec":lab,"treatment":treat,"theta":round(res.theta,4),
                         "se":round(res.se,4),"p_value":round(res.p_value,4),
                         "signif":sig,"n_obs":res.n_obs})
    except Exception as e:
        print(f"  {lab}: ERROR {e}")

df_dml = pd.DataFrame(rows_dml)
df_dml.to_csv(TABLES_DIR / "tableMS3_contamination.csv", index=False)
print(f"\nSaved: tableMS3_contamination.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PER-COUNTRY GPRC x Stress (DML)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("MS3-3. PER-COUNTRY GPRC x Stress (DML, + persistence control)")
print("="*65)
COUNTRY_LABEL = {"VNM":"Vietnam","THA":"Thailand","IDN":"Indonesia",
                 "MYS":"Malaysia","PHL":"Philippines"}
rows_cc = []
for cc in CC5:
    sub = panel[panel["country"]==cc].copy()
    ctrl = ["gpr_country_z", STRESS, "log_rv_t_z"] + BASE_CTRL
    df = sub.dropna(subset=[OUTCOME,"gpr_x_stress"]+ctrl)
    if len(df) < 40:
        print(f"  {COUNTRY_LABEL[cc]}: n={len(df)} skip"); continue
    try:
        nf = min(5, max(2, len(df)//12))
        est = DMLEstimator(ml_g="lasso", ml_m="lasso", n_folds=nf, n_reps=3, seed=42)
        res = est.fit(Y=df[OUTCOME].values, D=df["gpr_x_stress"].values,
                      X=df[ctrl].values, outcome=OUTCOME,
                      treatment="gpr_x_stress", country=cc)
        sig = "***" if res.p_value<0.01 else "**" if res.p_value<0.05 else "*" if res.p_value<0.10 else ""
        print(f"  {COUNTRY_LABEL[cc]:12s}: theta={res.theta:+.4f}  p={res.p_value:.3f} {sig}  n={res.n_obs}")
        rows_cc.append({"country":cc,"country_label":COUNTRY_LABEL[cc],
                        "theta":round(res.theta,4),"se":round(res.se,4),
                        "p_value":round(res.p_value,4),"signif":sig,
                        "ci_lower":round(res.ci_lower,4),"ci_upper":round(res.ci_upper,4),
                        "n_obs":res.n_obs})
    except Exception as e:
        print(f"  {cc}: ERROR {e}")
df_cc = pd.DataFrame(rows_cc)
if not df_cc.empty:
    df_cc.to_csv(TABLES_DIR / "tableMS3_by_country.csv", index=False)
    print("Saved: tableMS3_by_country.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: contamination test
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: b3 without vs with persistence (OLS)
ax0 = axes[0]
specs = ["WITHOUT logRV_t","WITH logRV_t"]
b3 = [float(df_int[(df_int["spec"]==s)&(df_int["term"]=="GPRC x Stress (b3)")]["beta"].iloc[0])
      for s in specs]
b3se = [float(df_int[(df_int["spec"]==s)&(df_int["term"]=="GPRC x Stress (b3)")]["se"].iloc[0])
        for s in specs]
b3p = [float(df_int[(df_int["spec"]==s)&(df_int["term"]=="GPRC x Stress (b3)")]["p_value"].iloc[0])
       for s in specs]
cols = ["#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0" for p in b3p]
spec_labels = ["Without\npersistence control", "With\npersistence control"]
ax0.bar(spec_labels, b3, yerr=[1.96*s for s in b3se], color=cols, alpha=0.85, capsize=6)
for i,(v,p) in enumerate(zip(b3,b3p)):
    s = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "n.s."
    ax0.text(i, v + (0.01 if v>=0 else -0.03), s, ha="center", fontweight="bold")
ax0.axhline(0, color="k", lw=0.8)
ax0.set_ylabel("β₃ (GPRC × Stress, std)")
ax0.set_title("With and without persistence control", fontweight="bold")
ax0.grid(axis="y", alpha=0.3)

# Right: per country GPRC x Stress (DML)
ax1 = axes[1]
if not df_cc.empty:
    sub = df_cc.set_index("country").reindex(CC5).reset_index().dropna(subset=["theta"])
    for i, row in sub.iterrows():
        p = row["p_value"]
        col = "#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0"
        ax1.errorbar(row["theta"], i,
                     xerr=[[row["theta"]-row["ci_lower"]],[row["ci_upper"]-row["theta"]]],
                     fmt="o", color=col, capsize=4, lw=2, ms=8)
        if row["signif"]:
            ax1.text(row["ci_upper"]+0.02, i, row["signif"], va="center", fontsize=10, color=col)
    ax1.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax1.set_yticks(range(len(sub)))
    ax1.set_yticklabels([COUNTRY_LABEL[c] for c in sub["country"]], fontsize=9)
    ax1.set_xlabel("GPRC × Stress (DML)")
    ax1.set_title("Country-level interaction estimates", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS_DIR / "figMS3_contamination.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGS_DIR / "figMS3_contamination.pdf", bbox_inches="tight")  # vector PDF
plt.close()
print(f"Saved: figMS3_contamination.png")

print("\nMS3 HOAN THANH.")
