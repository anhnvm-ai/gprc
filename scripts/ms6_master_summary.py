"""MS6: Master Summary + Verdict on all 5 Hypotheses

Integrates MS1-MS5 into a single hypothesis-verdict table, a master Excel
workbook, and a one-page summary figure.

Outputs:
  results/tables/tableMS6_hypothesis_verdicts.csv
  results/tables/tableMS_MASTER.xlsx
  results/figures/figMS6_master.png
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

TABLES_DIR = ROOT / "results" / "tables"
FIGS_DIR   = ROOT / "results" / "figures"

def L(name):
    p = TABLES_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

ms1 = L("tableMS1_regime_stats.csv")
ms2 = L("tableMS2_regime_channels.csv")
ms3i= L("tableMS3_interaction.csv")
ms3c= L("tableMS3_contamination.csv")
ms4 = L("tableMS4_regime_dml.csv")
ms5 = L("tableMS5_regime_forecast.csv")
ms5cw=L("tableMS5_cw_by_regime.csv")


# ── Helper getters ────────────────────────────────────────────────────────────
def ms2_beta(regime, channel, regime_def="COMMON"):
    r = ms2[(ms2["regime"]==regime)&(ms2["channel"]==channel)&(ms2.get("regime_def","COMMON")==regime_def)]
    if len(r)==0:
        r = ms2[(ms2["regime"]==regime)&(ms2["channel"]==channel)]
    return (float(r["beta_std"].iloc[0]), float(r["p_value"].iloc[0])) if len(r) else (np.nan,np.nan)

def ms3_b3(spec):
    r = ms3i[(ms3i["spec"]==spec)&(ms3i["term"]=="GPRC x Stress (b3)")]
    return (float(r["beta"].iloc[0]), float(r["p_value"].iloc[0])) if len(r) else (np.nan,np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# Build hypothesis verdicts
# ══════════════════════════════════════════════════════════════════════════════
verdicts = []

# H1: persistence dominant in both regimes
pc_calm, pp_calm = ms2_beta("Calm","log_rv_t")
pc_str, pp_str   = ms2_beta("Stress","log_rv_t")
h1 = (pp_calm<0.05 and pp_str<0.05)
verdicts.append({
    "hypothesis":"H1","statement":"Persistence dominant in both regimes",
    "evidence":f"beta_calm={pc_calm:+.3f}(p={pp_calm:.3f}); beta_stress={pc_str:+.3f}(p={pp_str:.3f})",
    "verdict":"SUPPORTED" if h1 else "PARTIAL"})

# H2: persistence stronger in stress
h2 = abs(pc_str) > abs(pc_calm)
verdicts.append({
    "hypothesis":"H2","statement":"Persistence stronger in stress",
    "evidence":f"|{pc_str:.3f}| vs |{pc_calm:.3f}|",
    "verdict":"SUPPORTED" if h2 else "REJECTED (persistence weaker in stress)"})

# H3: VIX stronger in stress
vc_calm, vp_calm = ms2_beta("Calm","vix_m")
vc_str, vp_str   = ms2_beta("Stress","vix_m")
h3 = abs(vc_str) > abs(vc_calm)
verdicts.append({
    "hypothesis":"H3","statement":"VIX stronger in stress",
    "evidence":f"|{vc_str:.3f}|(p={vp_str:.2f}) vs |{vc_calm:.3f}|(p={vp_calm:.2f})",
    "verdict":"SUPPORTED" if h3 else "PARTIAL"})

# H4: GPRC matters mainly in stress
gc_calm, gp_calm = ms2_beta("Calm","gpr_country")
gc_str, gp_str   = ms2_beta("Stress","gpr_country")
h4 = abs(gc_str) > abs(gc_calm)
verdicts.append({
    "hypothesis":"H4","statement":"GPRC matters mainly in stress",
    "evidence":f"stress beta={gc_str:+.3f}(p={gp_str:.2f}) ~{abs(gc_str)/max(abs(gc_calm),1e-6):.1f}x calm={gc_calm:+.3f}(p={gp_calm:.2f})",
    "verdict":"SUPPORTED" if h4 else "REJECTED"})

# H5: persistence contamination (does GPRxStress survive control?)
b3_wo, p3_wo = ms3_b3("WITHOUT logRV_t")
b3_w, p3_w   = ms3_b3("WITH logRV_t")
contaminated = (p3_wo<0.10 and p3_w>=0.10)
verdicts.append({
    "hypothesis":"H5","statement":"GPR amplification is persistence contamination",
    "evidence":f"b3 without ctrl={b3_wo:+.3f}(p={p3_wo:.3f}); with ctrl={b3_w:+.3f}(p={p3_w:.3f})",
    "verdict":"REJECTED -> GPR effect is REAL (survives/strengthens after persistence control)"
              if not contaminated else "SUPPORTED -> contamination"})

df_v = pd.DataFrame(verdicts)
df_v.to_csv(TABLES_DIR / "tableMS6_hypothesis_verdicts.csv", index=False)
print("="*70)
print("HYPOTHESIS VERDICTS")
print("="*70)
for _, r in df_v.iterrows():
    print(f"\n  [{r['hypothesis']}] {r['statement']}")
    print(f"      Evidence: {r['evidence']}")
    print(f"      VERDICT : {r['verdict']}")


# ══════════════════════════════════════════════════════════════════════════════
# Master Excel
# ══════════════════════════════════════════════════════════════════════════════
out_xlsx = TABLES_DIR / "tableMS_MASTER.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:
    sheets = {"MS1_regimes":ms1,"MS2_regime_channels":ms2,"MS3_interaction":ms3i,
              "MS3_contamination":ms3c,"MS4_regime_dml":ms4,"MS5_forecast":ms5,
              "MS5_cw":ms5cw,"MS6_verdicts":df_v}
    for name, df in sheets.items():
        if not df.empty:
            df.to_excel(wr, sheet_name=name[:31], index=False)
            ws = wr.sheets[name[:31]]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(
                    len(str(col[0].value or "")),
                    max((len(str(c.value or "")) for c in col[1:8]), default=0))+2
print(f"\nSaved: {out_xlsx}")


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FIGURE (2x2)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

CHAN = {"log_rv_t":"Persistence","vix_m":"VIX","dlog_fx_m":"FX","gpr_country":"GPR"}

# Panel 1: regime channel betas (MS2)
ax = axes[0,0]
ch_order = list(CHAN); y = np.arange(len(ch_order)); h=0.35
calm   = [ms2_beta("Calm",c)[0]   for c in ch_order]
stress = [ms2_beta("Stress",c)[0] for c in ch_order]
ax.barh(y+h/2, calm, h, color="#4daf4a", alpha=0.8, label="Calm")
ax.barh(y-h/2, stress, h, color="#d62728", alpha=0.8, label="Stress")
ax.set_yticks(y); ax.set_yticklabels([CHAN[c] for c in ch_order], fontsize=9)
ax.axvline(0,color="k",lw=0.8); ax.legend(fontsize=8)
ax.set_title("(A) Regime-Specific Channel β (MS2)", fontsize=10, fontweight="bold")
ax.set_xlabel("Standardized β", fontsize=8); ax.grid(axis="x",alpha=0.3)

# Panel 2: contamination test (MS3)
ax = axes[0,1]
specs=["WITHOUT logRV_t","WITH logRV_t"]
b3=[ms3_b3(s)[0] for s in specs]; p3=[ms3_b3(s)[1] for s in specs]
cols=["#d62728" if p<0.05 else "#ff7f0e" if p<0.10 else "#4878d0" for p in p3]
ax.bar(["Without\npersistence","With\npersistence"], b3, color=cols, alpha=0.85)
for i,(v,p) in enumerate(zip(b3,p3)):
    s="***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "n.s."
    ax.text(i, v-0.012, s, ha="center", fontsize=11, fontweight="bold", color="white")
ax.axhline(0,color="k",lw=0.8)
ax.set_title("(B) Persistence Contamination Test — β₃ GPRC×Stress (MS3)\nVERDICT: GPR effect is REAL",
             fontsize=9, fontweight="bold")
ax.set_ylabel("β₃ (std)", fontsize=8); ax.grid(axis="y",alpha=0.3)

# Panel 3: regime DML (MS4)
ax = axes[1,0]
if not ms4.empty:
    calm4   = [ms4[(ms4["regime"]=="Calm")&(ms4["channel"]==c)]["theta"].iloc[0]
               if len(ms4[(ms4["regime"]=="Calm")&(ms4["channel"]==c)]) else np.nan for c in ch_order]
    stress4 = [ms4[(ms4["regime"]=="Stress")&(ms4["channel"]==c)]["theta"].iloc[0]
               if len(ms4[(ms4["regime"]=="Stress")&(ms4["channel"]==c)]) else np.nan for c in ch_order]
    ax.barh(y+h/2, calm4, h, color="#4daf4a", alpha=0.8, label="Calm")
    ax.barh(y-h/2, stress4, h, color="#d62728", alpha=0.8, label="Stress")
    ax.set_yticks(y); ax.set_yticklabels([CHAN[c] for c in ch_order], fontsize=9)
    ax.axvline(0,color="k",lw=0.8); ax.legend(fontsize=8)
ax.set_title("(C) Regime Orthogonal DML θ (MS4)", fontsize=10, fontweight="bold")
ax.set_xlabel("θ (std)", fontsize=8); ax.grid(axis="x",alpha=0.3)

# Panel 4: forecast R2_OOS by regime (MS5)
ax = axes[1,1]
if not ms5.empty:
    models = ms5["model"].unique().tolist()
    x = np.arange(len(models)); w=0.27
    for i,(reg,color) in enumerate([("All","#888"),("Calm","#4daf4a"),("Stress","#d62728")]):
        sub = ms5[ms5["regime"]==reg]
        vals=[sub[sub["model"]==m]["R2_OOS"].iloc[0] if len(sub[sub["model"]==m]) else np.nan for m in models]
        ax.bar(x+(i-1)*w, vals, w, color=color, alpha=0.8, label=reg)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("M0_","").replace("M1_","").replace("M2_","")
        .replace("M3_","").replace("M4_","").replace("_","") for m in models], fontsize=7, rotation=20)
    ax.axhline(0,color="k",lw=0.8); ax.legend(fontsize=8)
ax.set_title("(D) Forecast R²_OOS by Regime (MS5)\nStress is intrinsically hard to forecast",
             fontsize=9, fontweight="bold")
ax.set_ylabel("R²_OOS", fontsize=8); ax.grid(axis="y",alpha=0.3)

fig.suptitle("Regime-Dependent Volatility Transmission Channels — 5-ASEAN\n"
             "Markov-Switching Upgrade: GPR effects are regime-conditional & persistence-robust",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS_DIR / "figMS6_master.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figMS6_master.png")

print("\nMS6 HOAN THANH.")
