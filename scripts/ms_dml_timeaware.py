"""MS-DML-TIMEAWARE: cross-fitting scheme robustness for the DML channel ranking.

The DML channel estimates in the paper are produced with conventional random
K-fold cross-fitting (cv_scheme="shuffled", median over 3 seeds). Random folds
can, in autocorrelated monthly panels, place future months in the training set
used to residualise a past month (look-ahead leakage). This script re-estimates
the same pooled specification with TIME-AWARE blocked cross-fitting
(cv_scheme="blocked"): each validation fold is a contiguous block of calendar
months and the training set is restricted to strictly-earlier months, so
training observations always precede validation observations.

It compares, for the four channels and for the GPRC x Stress interaction:
    shuffled (baseline)  vs  blocked (time-aware)
reporting theta, SE, p-value and effective n. The claim being defended is
qualitative: the ranking of channels and the sign of the interaction do not
depend on the cross-fitting scheme.

Outputs:
  results/tables/tableMS_dml_timeaware.csv   (long form)
  results/tables/tableMS_dml_timeaware_tex.txt (appendix LaTeX table body)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.utils.io import ROOT
from src.models.dml import DMLEstimator

PANEL_PATH   = ROOT / "data" / "processed" / "asean_panel_monthly.parquet"
REGIME_PATH  = ROOT / "data" / "processed" / "asean_panel_regimes.parquet"
TABLES_DIR   = ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CC5 = ["VNM", "THA", "IDN", "MYS", "PHL"]
ML  = "lasso"          # matches the baseline channel figure (Lasso pooled)
SEED = 42

CHANNELS = {
    "log_rv_lag1": "Volatility persistence",
    "gpr_country": "GPRC",
    "vix_m":       "VIX (global risk)",
    "dlog_fx_m":   "FX shock",
}
MACRO_CONTROLS = ["sp500_ret_m", "us10y_m", "wti_ret_m", "ret_m"]
OUTCOME = "log_rv_lead1"


def zscore_within_country(panel, cols):
    for col in cols:
        if col in panel.columns:
            panel[col + "_z"] = panel.groupby("country")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-12))
    return panel


def run_one(df, outcome_z, treat_z, ctrl_cols, scheme):
    """Fit pooled DML under a given cv_scheme; return a result dict or None."""
    ctrl = [c for c in ctrl_cols if c in df.columns and df[c].notna().sum() > 100]
    d = df.dropna(subset=[outcome_z, treat_z] + ctrl)
    if len(d) < 50:
        return None
    n_reps = 3 if scheme == "shuffled" else 1
    est = DMLEstimator(ml_g=ML, ml_m=ML, n_folds=5, n_reps=n_reps,
                       seed=SEED, cv_scheme=scheme)
    res = est.fit(Y=d[outcome_z].values, D=d[treat_z].values,
                  X=d[ctrl].values, outcome=outcome_z, treatment=treat_z,
                  country="POOLED", dates=d.index.values)
    sig = "***" if res.p_value < 0.01 else "**" if res.p_value < 0.05 \
          else "*" if res.p_value < 0.10 else ""
    return {"theta": res.theta, "se": res.se, "p_value": res.p_value,
            "n_obs": res.n_obs, "signif": sig}


# ══════════════════════════════════════════════════════════════════════════════
# Load panels
# ══════════════════════════════════════════════════════════════════════════════
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index("date").sort_index()
panel = panel[panel["country"].isin(CC5)].copy()
panel["log_rv"]       = np.log(panel["rv_m"].clip(lower=1e-8))
panel["log_rv_lead1"] = panel.groupby("country")["log_rv"].shift(-1)
panel["log_rv_lag1"]  = panel["log_rv"]

panel = zscore_within_country(panel, list(CHANNELS.keys()) + MACRO_CONTROLS + [OUTCOME])
ch_z      = {k: k + "_z" for k in CHANNELS}
macro_z   = [c + "_z" for c in MACRO_CONTROLS if c in panel.columns]
outcome_z = OUTCOME + "_z"

# country dummies for the pooled spec
cty_dummies = pd.get_dummies(panel["country"], prefix="cty", drop_first=True)
cty_cols    = list(cty_dummies.columns)
panel_pool  = pd.concat([panel.reset_index(), cty_dummies.reset_index(drop=True)], axis=1)
panel_pool  = panel_pool.set_index("date").sort_index()

print("=" * 70)
print("DML CROSS-FITTING ROBUSTNESS: shuffled (baseline) vs blocked (time-aware)")
print("=" * 70)

rows = []

# ── Four channels ─────────────────────────────────────────────────────────────
print("\n--- Pooled channels (Lasso) ---")
print(f"  {'Channel':24s} {'scheme':9s} {'theta':>9s} {'SE':>8s} {'p':>7s}  n")
for treat_col, label in CHANNELS.items():
    treat_z   = treat_col + "_z"
    other_z   = [ch_z[c] for c in CHANNELS if c != treat_col]
    ctrl_pool = other_z + macro_z + cty_cols
    for scheme in ["shuffled", "blocked"]:
        r = run_one(panel_pool, outcome_z, treat_z, ctrl_pool, scheme)
        if r is None:
            print(f"  {label:24s} {scheme:9s}  (skipped)")
            continue
        print(f"  {label:24s} {scheme:9s} {r['theta']:+9.4f} {r['se']:8.4f} "
              f"{r['p_value']:7.3f}{r['signif']:3s} {r['n_obs']}")
        rows.append({"term": label, "scheme": scheme, **r})


# ── GPRC x Stress interaction ─────────────────────────────────────────────────
print("\n--- GPRC x Stress interaction (Lasso) ---")
reg = pd.read_parquet(REGIME_PATH)
reg["date"] = pd.to_datetime(reg["date"])
reg = reg.set_index("date").sort_index()
reg = reg[reg["country"].isin(CC5)].copy()
reg["log_rv"]       = np.log(reg["rv_m"].clip(lower=1e-8))
reg["log_rv_lead1"] = reg.groupby("country")["log_rv"].shift(-1)
reg["log_rv_t"]     = reg["log_rv"]

reg = zscore_within_country(
    reg, ["gpr_country", "vix_m", "dlog_fx_m", "log_rv_t",
          "sp500_ret_m", "us10y_m", "wti_ret_m", "ret_m", "log_rv_lead1"])
reg["gpr_x_stress"] = reg["gpr_country_z"] * reg["stress_common"]

cty_d2 = pd.get_dummies(reg["country"], prefix="cty", drop_first=True)
reg_pool = pd.concat([reg.reset_index(), cty_d2.reset_index(drop=True)], axis=1)
reg_pool = reg_pool.set_index("date").sort_index()

INT_CTRL = (["gpr_country_z", "stress_common", "log_rv_t_z", "vix_m_z",
             "dlog_fx_m_z", "sp500_ret_m_z", "us10y_m_z", "wti_ret_m_z", "ret_m_z"]
            + list(cty_d2.columns))

print(f"  {'term':24s} {'scheme':9s} {'theta':>9s} {'SE':>8s} {'p':>7s}  n")
for scheme in ["shuffled", "blocked"]:
    r = run_one(reg_pool, "log_rv_lead1_z", "gpr_x_stress", INT_CTRL, scheme)
    if r is None:
        print(f"  {'GPRC x Stress':24s} {scheme:9s}  (skipped)")
        continue
    print(f"  {'GPRC x Stress':24s} {scheme:9s} {r['theta']:+9.4f} {r['se']:8.4f} "
          f"{r['p_value']:7.3f}{r['signif']:3s} {r['n_obs']}")
    rows.append({"term": "GPRC x Stress", "scheme": scheme, **r})


# ══════════════════════════════════════════════════════════════════════════════
# Save long table
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(rows)
for c in ["theta", "se", "p_value"]:
    df[c] = df[c].round(4)
df.to_csv(TABLES_DIR / "tableMS_dml_timeaware.csv", index=False)
print(f"\nSaved: tableMS_dml_timeaware.csv")


# ── Wide form -> LaTeX body for the appendix ──────────────────────────────────
order = ["Volatility persistence", "VIX (global risk)", "FX shock", "GPRC",
         "GPRC x Stress"]
piv = df.pivot(index="term", columns="scheme",
               values=["theta", "se", "p_value", "n_obs"])
lines = []
for term in order:
    if term not in piv.index:
        continue
    sh_t = piv.loc[term, ("theta", "shuffled")]
    sh_p = piv.loc[term, ("p_value", "shuffled")]
    bl_t = piv.loc[term, ("theta", "blocked")]
    bl_p = piv.loc[term, ("p_value", "blocked")]
    bl_n = piv.loc[term, ("n_obs", "blocked")]
    tname = term.replace("GPRC x Stress", r"GPRC $\times$ Stress")
    lines.append(f"{tname} & {sh_t:+.3f} & {sh_p:.3f} & {bl_t:+.3f} & {bl_p:.3f} & {int(bl_n)} \\\\")
tex = "\n".join(lines)
(TABLES_DIR / "tableMS_dml_timeaware_tex.txt").write_text(tex, encoding="utf-8")
print("Saved: tableMS_dml_timeaware_tex.txt")
print("\nLaTeX body:\n" + tex)

# ── Quick verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
w = df.pivot(index="term", columns="scheme", values="theta")
for term in w.index:
    if {"shuffled", "blocked"}.issubset(w.columns):
        s, b = w.loc[term, "shuffled"], w.loc[term, "blocked"]
        same_sign = "same sign" if np.sign(s) == np.sign(b) else "SIGN FLIP"
        print(f"  {term:24s}: shuffled={s:+.4f}  blocked={b:+.4f}  [{same_sign}]")
print("\nDONE.")
