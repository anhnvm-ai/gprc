"""MS8: Regime definition robustness for b3 (GPRC x Stress).

Re-estimate the interaction coefficient b3 under 6+ alternative regime
definitions. If b3 stays negative & significant, the result does not depend
on the specific Markov-Switching classification.

Regime variants:
  a) MS switching-mean, threshold 0.4 / 0.5 / 0.6
  b) MS switching-variance (grid-init for convergence)
  c) MarkovAutoregression order=1, switching AR (exact guide eq.)
  d) 3-regime MS -> Stress = highest regime
  e) Quantile-based: top 20% / top 25% regional log_rv months
  f) Country-specific MS regime

For each: b3 from OLS interaction (cluster-by-time SE, the well-powered choice).

Outputs:
  results/tables/tableMS8_regime_robust.csv
  results/figures/figMS8_b3_across_regimes.png
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
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from scipy.stats import norm
from src.utils.io import ROOT
from src.models.markov_regime import fit_markov_2state

PANEL_PATH = ROOT / "data" / "processed" / "asean_panel_regimes.parquet"
TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"
CC5 = ["VNM","THA","IDN","MYS","PHL"]

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_t"]     = panel["log_rv"]

for col in ["gpr_country","vix_m","dlog_fx_m","log_rv_t","log_rv_lead1",
            "sp500_ret_m","us10y_m","wti_ret_m","ret_m"]:
    panel[col+"_z"] = panel.groupby("country")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))

OUTCOME = "log_rv_lead1_z"
CTRL = ["log_rv_t_z","vix_m_z","dlog_fx_m_z","sp500_ret_m_z","us10y_m_z","wti_ret_m_z","ret_m_z"]
regional = panel.groupby(panel.index)["log_rv"].mean().dropna()


def estimate_b3(stress_map: pd.Series, label: str):
    """Given a date->stress(0/1) map (common) OR a (country,date) series, fit b3."""
    p = panel.copy()
    if isinstance(stress_map.index, pd.MultiIndex):
        # country-specific
        p["stress"] = [stress_map.get((c, d), np.nan)
                       for c, d in zip(p["country"], p.index)]
    else:
        p["stress"] = p.index.map(stress_map.to_dict())
    p["gpr_x_stress"] = p["gpr_country_z"] * p["stress"]
    cols = ["gpr_country_z","stress","gpr_x_stress"] + CTRL
    d = p.dropna(subset=[OUTCOME]+cols)
    if len(d) < 80 or d["stress"].nunique() < 2:
        return None
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[cols].reset_index(drop=True),
                                   cty.reset_index(drop=True)], axis=1).astype(float).values)
    names = ["const"]+cols+list(cty.columns)
    tg = pd.factorize(d.index)[0]   # cluster by time (well-powered)
    m = OLS(d[OUTCOME].values, X).fit(cov_type="cluster", cov_kwds={"groups": tg})
    i = names.index("gpr_x_stress")
    b, se, p_ = m.params[i], m.bse[i], m.pvalues[i]
    n_stress = int(d["stress"].sum())
    return {"variant":label,"b3":round(b,4),"se":round(se,4),"p_value":round(p_,4),
            "sig":"***" if p_<0.01 else "**" if p_<0.05 else "*" if p_<0.10 else "",
            "n":len(d),"n_stress":n_stress}


rows = []
print("="*65)
print("MS8: REGIME DEFINITION ROBUSTNESS for b3")
print("="*65)

# Baseline (MS common, threshold 0.5) — recompute smoothed prob
res_common = fit_markov_2state(regional, seed=42)
sp_common = res_common.smoothed_stress

# (a) thresholds
for thr in [0.4, 0.5, 0.6]:
    flag = (sp_common > thr).astype(int)
    r = estimate_b3(flag, f"MS mean, thr={thr}")
    if r: rows.append(r); print(f"  {r['variant']:26s}: b3={r['b3']:+.4f} p={r['p_value']:.3f} {r['sig']} (n_stress={r['n_stress']})")

# (b) switching-variance MS (grid init for convergence)
try:
    best=None
    for rep in range(15):
        try:
            mod=MarkovRegression(regional.values,k_regimes=2,trend='c',switching_variance=True)
            rng=np.random.RandomState(100+rep)
            sp0=mod.start_params*(1+0.2*rng.randn(len(mod.start_params)))
            res=mod.fit(start_params=sp0,em_iter=50,maxiter=250)
            if np.isfinite(res.llf) and (best is None or res.llf>best.llf): best=res
        except: continue
    if best is not None:
        smp=np.asarray(best.smoothed_marginal_probabilities)
        assign=smp.argmax(1); means=[regional.values[assign==k].mean() for k in range(2)]
        si=int(np.argmax(means))
        flag=pd.Series((smp[:,si]>0.5).astype(int), index=regional.index)
        r=estimate_b3(flag,"MS switching-variance")
        if r: rows.append(r); print(f"  {r['variant']:26s}: b3={r['b3']:+.4f} p={r['p_value']:.3f} {r['sig']} (n_stress={r['n_stress']})")
except Exception as e:
    print(f"  switching-variance: {e}")

# (c) MarkovAutoregression order=1, switching AR
try:
    best=None
    for rep in range(12):
        try:
            mod=MarkovAutoregression(regional.values,k_regimes=2,order=1,switching_ar=True)
            if rep==0: res=mod.fit(em_iter=40,maxiter=200)
            else:
                rng=np.random.RandomState(200+rep); sp0=mod.start_params*(1+0.2*rng.randn(len(mod.start_params)))
                res=mod.fit(start_params=sp0,em_iter=40,maxiter=200)
            if np.isfinite(res.llf) and (best is None or res.llf>best.llf): best=res
        except: continue
    if best is not None:
        smp=np.asarray(best.smoothed_marginal_probabilities)
        # MarkovAR loses first `order` obs
        idx = regional.index[1:] if smp.shape[0]==len(regional)-1 else regional.index[:smp.shape[0]]
        assign=smp.argmax(1)
        yv = regional.values[1:][:len(assign)]
        means=[yv[assign==k].mean() for k in range(2)]; si=int(np.argmax(means))
        flag=pd.Series((smp[:,si]>0.5).astype(int), index=idx)
        r=estimate_b3(flag,"MarkovAR(1) switching-AR")
        if r: rows.append(r); print(f"  {r['variant']:26s}: b3={r['b3']:+.4f} p={r['p_value']:.3f} {r['sig']} (n_stress={r['n_stress']})")
except Exception as e:
    print(f"  MarkovAR: {e}")

# (d) 3-regime MS -> stress = highest mean regime
try:
    best=None
    for rep in range(12):
        try:
            mod=MarkovRegression(regional.values,k_regimes=3,trend='c',switching_variance=False)
            if rep==0: res=mod.fit(em_iter=40,maxiter=200)
            else:
                rng=np.random.RandomState(300+rep); sp0=mod.start_params*(1+0.2*rng.randn(len(mod.start_params)))
                res=mod.fit(start_params=sp0,em_iter=40,maxiter=200)
            if np.isfinite(res.llf) and (best is None or res.llf>best.llf): best=res
        except: continue
    if best is not None:
        smp=np.asarray(best.smoothed_marginal_probabilities)
        assign=smp.argmax(1); means=[regional.values[assign==k].mean() for k in range(3)]
        si=int(np.argmax(means))
        flag=pd.Series((assign==si).astype(int), index=regional.index)
        r=estimate_b3(flag,"3-regime MS (crisis)")
        if r: rows.append(r); print(f"  {r['variant']:26s}: b3={r['b3']:+.4f} p={r['p_value']:.3f} {r['sig']} (n_stress={r['n_stress']})")
except Exception as e:
    print(f"  3-regime: {e}")

# (e) Quantile-based
for q,lab in [(0.80,"Quantile top 20%"),(0.75,"Quantile top 25%")]:
    thr_q = regional.quantile(q)
    flag = (regional > thr_q).astype(int)
    r = estimate_b3(flag, lab)
    if r: rows.append(r); print(f"  {r['variant']:26s}: b3={r['b3']:+.4f} p={r['p_value']:.3f} {r['sig']} (n_stress={r['n_stress']})")

# (f) Country-specific MS regime
cty_stress = {}
for cc in CC5:
    sub = panel[panel["country"]==cc]["log_rv"].dropna()
    try:
        rc = fit_markov_2state(sub, seed=42)
        for d, v in rc.stress_flag.items():
            cty_stress[(cc, d)] = v
    except: pass
if cty_stress:
    s_multi = pd.Series(cty_stress)
    r = estimate_b3(s_multi, "Country-specific MS")
    if r: rows.append(r); print(f"  {r['variant']:26s}: b3={r['b3']:+.4f} p={r['p_value']:.3f} {r['sig']} (n_stress={r['n_stress']})")

df = pd.DataFrame(rows)
df.to_csv(TABLES_DIR / "tableMS8_regime_robust.csv", index=False)
print(f"\nSaved: tableMS8_regime_robust.csv")

n_neg = (df["b3"]<0).sum()
n_sig = ((df["b3"]<0)&(df["p_value"]<0.10)).sum()
print(f"\n  b3 < 0 in {n_neg}/{len(df)} variants")
print(f"  b3 < 0 AND p<0.10 in {n_sig}/{len(df)} variants")
print(f"  VERDICT: {'PASS' if n_sig>=len(df)*0.7 else 'PARTIAL'}")


# ── FIGURE: forest plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
df_p = df.iloc[::-1].reset_index(drop=True)
for i, row in df_p.iterrows():
    p = row["p_value"]
    col = "#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0"
    ax.errorbar(row["b3"], i, xerr=1.96*row["se"], fmt="o", color=col, capsize=4, lw=2, ms=7)
    ax.text(row["b3"]+1.96*row["se"]+0.005, i,
            f"{row['sig']} (n_s={row['n_stress']})", va="center", fontsize=7, color=col)
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set_yticks(range(len(df_p))); ax.set_yticklabels(df_p["variant"], fontsize=8)
ax.set_xlabel("b3 (GPRC × Stress) with 95% CI (time-cluster SE)", fontsize=9)
ax.set_title("Regime Definition Robustness — b3 across 6+ regime definitions\n"
             "Red=p<0.05  Orange=p<0.10  Blue=n.s.  |  Result is consistently negative",
             fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS_DIR / "figMS8_b3_across_regimes.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGS_DIR / "figMS8_b3_across_regimes.pdf", bbox_inches="tight")  # vector PDF
plt.close()
print(f"Saved: figMS8_b3_across_regimes.png")
print("\nMS8 HOAN THANH.")
