"""Step A3/A3b: collect every number that appears in the paper
from the shadow-run tables (cr1) and the extra analyses (cr3) into one flat
JSON, and compare against the submitted PDF.

  python scripts/cr2_collect.py raw   -> results/numbers_raw.json + results/checksum_raw.md
  python scripts/cr2_collect.py cr    -> results/numbers_cr.json  + results/diff_raw_vs_cr.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crutil import CR, RES, save_json, load_json  # noqa: E402

CC5 = ["VNM", "THA", "IDN", "MYS", "PHL"]
CH = {"log_rv_lag1": "persistence", "vix_m": "vix", "dlog_fx_m": "fx", "gpr_country": "gprc"}

# Values printed in the submitted PDF: key -> (value, decimals shown)
PAPER = {
    "t2.log_rv.mean": (-2.054, 3), "t2.log_rv.std": (0.467, 3), "t2.log_rv.min": (-3.377, 3),
    "t2.log_rv.median": (-2.070, 3), "t2.log_rv.max": (-0.186, 3),
    "t2.gprc.mean": (0.041, 3), "t2.gprc.std": (0.048, 3), "t2.gprc.median": (0.030, 3),
    "t2.gprc.max": (0.973, 3), "t2.vix.mean": (18.31, 2), "t2.vix.std": (6.21, 2),
    "t2.vix.min": (10.13, 2), "t2.vix.median": (16.91, 2), "t2.vix.max": (57.74, 2),
    "t2.dlog_fx.mean": (0.001, 3), "t2.dlog_fx.std": (0.022, 3), "t2.dlog_fx.min": (-0.148, 3),
    "t2.dlog_fx.median": (0.001, 3), "t2.dlog_fx.max": (0.240, 3),
    "t2.sp500.mean": (0.010, 3), "t2.sp500.std": (0.041, 3), "t2.sp500.min": (-0.134, 3),
    "t2.sp500.median": (0.017, 3), "t2.sp500.max": (0.119, 3),
    "t2.domestic_ret.mean": (0.004, 3), "t2.domestic_ret.std": (0.045, 3),
    "t2.domestic_ret.min": (-0.286, 3), "t2.domestic_ret.median": (0.007, 3),
    "t2.domestic_ret.max": (0.164, 3),
    "s41.vix_corr": (0.336, 3), "s41.persist.VNM": (0.22, 2), "s41.persist.THA": (0.47, 2),
    "t3.fe.persistence": (0.341, 3), "t3.fe.vix": (-0.181, 3), "t3.fe.fx": (0.035, 3), "t3.fe.gprc": (-0.023, 3),
    "t3.dml.persistence": (0.467, 3), "t3.dml.vix": (0.068, 3), "t3.dml.fx": (0.044, 3), "t3.dml.gprc": (-0.053, 3),
    "t3.dml_p.vix": (0.044, 3), "t3.dml_p.gprc": (0.049, 3),
    "t3.dr2.persistence": (16.0, 1), "t3.dr2.vix": (2.7, 1), "t3.dr2.fx": (0.3, 1), "t3.dr2.gprc": (0.1, 1),
    "t3.r2full": (0.390, 3),
    "t3.cty.VNM.persistence": (0.396, 3), "t3.cty.THA.persistence": (0.668, 3), "t3.cty.IDN.persistence": (0.498, 3),
    "t3.cty.MYS.persistence": (0.388, 3), "t3.cty.PHL.persistence": (0.323, 3),
    "t3.cty.VNM.vix": (0.181, 3), "t3.cty.THA.vix": (-0.050, 3), "t3.cty.IDN.vix": (-0.005, 3),
    "t3.cty.MYS.vix": (0.207, 3), "t3.cty.PHL.vix": (0.098, 3),
    "t3.cty.VNM.fx": (-0.071, 3), "t3.cty.THA.fx": (0.081, 3), "t3.cty.IDN.fx": (0.107, 3),
    "t3.cty.MYS.fx": (0.109, 3), "t3.cty.PHL.fx": (-0.036, 3),
    "t3.cty.VNM.gprc": (-0.130, 3), "t3.cty.THA.gprc": (-0.088, 3), "t3.cty.IDN.gprc": (-0.080, 3),
    "t3.cty.MYS.gprc": (-0.011, 3), "t3.cty.PHL.gprc": (-0.108, 3),
    "t3.blocked.persistence": (0.489, 3), "t3.blocked.vix": (0.147, 3),
    "t3.blocked.fx": (0.045, 3), "t3.blocked.gprc": (-0.084, 3),
    "reg.n_stress": (38, 0), "reg.n_total": (196, 0), "reg.dur_stress": (4.3, 1),
    "split.calm": (-0.034, 3), "split.stress": (-0.129, 3),
    "t4.ols_nop.b": (-0.109, 3), "t4.ols_nop.ci_l": (-0.265, 3), "t4.ols_nop.ci_u": (0.047, 3), "t4.ols_nop.p": (0.169, 3),
    "t4.ols_p.b": (-0.170, 3), "t4.ols_p.ci_l": (-0.328, 3), "t4.ols_p.ci_u": (-0.011, 3), "t4.ols_p.p": (0.036, 3),
    "t4.dml.b": (-0.152, 3), "t4.dml.ci_l": (-0.292, 3), "t4.dml.ci_u": (-0.012, 3), "t4.dml.p": (0.033, 3),
    "t4.blocked.b": (-0.199, 3), "t4.blocked.ci_l": (-0.356, 3), "t4.blocked.ci_u": (-0.042, 3),
    "t4.blocked.p": (0.013, 3), "t4.blocked.n": (780, 0),
    "onset.calm": (158, 0), "onset.events": (10, 0), "onset.ctrl_or": (0.96, 2),
    "onset.ctrl_or_l": (0.47, 2), "onset.ctrl_or_u": (1.97, 2), "onset.ctrl_p": (0.910, 3),
    "onset.only_auc": (0.46, 2), "onset.ctrl_auc": (0.69, 2),
    "oos.calm_r2": (0.37, 2), "oos.stress_r2": (-0.20, 2),
    "rob.dk_p": (0.032, 3), "rob.webb_p": (0.028, 3),
    "rob.reg8_negative": (8, 0), "rob.reg8_sig05_claimed": (4, 0),
    "plac.perm_one_sided": (0.023, 3), "plac.return_p": (0.240, 3),
    "nonlin.max_p": (0.09, 2),
    "fx.b3_corrected": (-0.171, 3),
}


def tab(variant: str, name: str) -> pd.DataFrame:
    return pd.read_csv(CR / f"shadow_{variant}" / "results" / "tables" / name)


def collect(variant: str) -> dict:
    n: dict = {}
    ex = load_json(RES / f"extra_{variant}.json")

    # Table 2 + Sect. 4.1
    for var, st in ex["desc"]["table2"].items():
        for k in ["mean", "std", "min", "median", "max"]:
            n[f"t2.{var}.{k}"] = st[k]
    n["s41.vix_corr"] = ex["desc"]["corr_vix_rvlead_pooled_z"]
    n["s41.gprc_maxabs"] = ex["desc"]["corr_gprc_max_abs_z"]
    for c, v in ex["desc"]["corr_persistence_by_country"].items():
        n[f"s41.persist.{c}"] = v
    for c, v in ex["desc"]["corr_vix_rvlead_by_country"].items():
        n[f"s41.vixcorr.{c}"] = v

    # Table 3
    fe = tab(variant, "tableA1_fe_standardized.csv").set_index("channel")
    r2 = tab(variant, "tableA2_incremental_r2.csv").set_index("channel")
    dml = tab(variant, "tableB1_dml_channels_pooled.csv")
    dml = dml[dml["ml"] == "lasso"].set_index("channel")
    cty = tab(variant, "tableB2_dml_channels_by_country.csv")
    cty = cty[cty["ml"] == "lasso"]
    for ch, key in CH.items():
        n[f"t3.fe.{key}"] = fe.at[ch, "beta_std"]
        n[f"t3.fe_p.{key}"] = fe.at[ch, "p_value"]
        n[f"t3.dr2.{key}"] = r2.at[ch, "pct_contribution"]
        n[f"t3.dml.{key}"] = dml.at[ch, "theta"]
        n[f"t3.dml_p.{key}"] = dml.at[ch, "p_value"]
        for c in CC5:
            row = cty[(cty["country"] == c) & (cty["channel"] == ch)].iloc[0]
            n[f"t3.cty.{c}.{key}"] = row["theta"]
            n[f"t3.cty_p.{c}.{key}"] = row["p_value"]
    n["t3.r2full"] = r2.iloc[0]["r2_without"] + r2.iloc[0]["delta_r2"]
    ta = tab(variant, "tableMS_dml_timeaware.csv")
    lab = {"Volatility persistence": "persistence", "VIX (global risk)": "vix",
           "FX shock": "fx", "GPRC": "gprc", "GPRC x Stress": "gxs"}
    for r in ta.itertuples():
        k = lab[r.term]
        pre = "t3" if k != "gxs" else "t4"
        tag = "blocked" if r.scheme == "blocked" else "shuffled"
        n[f"{pre}.{tag}.{k}" if pre == "t3" else f"t4.{'blocked' if tag == 'blocked' else 'dml'}.b"] = r.theta
        if pre == "t3":
            n[f"t3.{tag}_p.{k}"] = r.p_value
            n[f"t3.{tag}_n.{k}"] = r.n_obs
        else:
            base = "t4.blocked" if tag == "blocked" else "t4.dml"
            n[f"{base}.se"] = r.se
            n[f"{base}.p"] = r.p_value
            n[f"{base}.n"] = r.n_obs
            n[f"{base}.ci_l"] = r.theta - 1.96 * r.se
            n[f"{base}.ci_u"] = r.theta + 1.96 * r.se

    # regimes and split-sample estimates
    ms1 = tab(variant, "tableMS1_regime_stats.csv")
    com = ms1[ms1["regime_def"] == "COMMON_ASEAN"].iloc[0]
    n["reg.n_stress"], n["reg.n_total"] = int(com["n_stress"]), int(com["n_total"])
    n["reg.dur_stress"], n["reg.pct_stress"] = com["dur_stress"], com["pct_stress"]
    ms2 = tab(variant, "tableMS2_regime_channels.csv")
    ms2 = ms2[(ms2["regime_def"] == "COMMON") & (ms2["channel"] == "gpr_country")].set_index("regime")
    n["split.calm"], n["split.stress"] = ms2.at["Calm", "beta_std"], ms2.at["Stress", "beta_std"]
    n["split.ratio"] = abs(n["split.stress"]) / abs(n["split.calm"])
    n["split.diff"] = abs(n["split.stress"] - n["split.calm"])

    # Table 4 OLS rows
    for tag, key in [("ols_nop", "no_persistence"), ("ols_p", "persistence")]:
        t = ex["table4"][key]
        n[f"t4.{tag}.b"] = t["b3"]
        n[f"t4.{tag}.ci_l"], n[f"t4.{tag}.ci_u"] = t["hc3"]["ci"]
        n[f"t4.{tag}.p"] = t["hc3"]["p"]
        n[f"t4.{tag}.n"] = t["n"]
        for s in ["time_cluster", "dk4", "dk6", "dk12"]:
            n[f"t4.{tag}.p_{s}"] = t[s]["p"]
            n[f"t4.{tag}.ci_{s}"] = t[s]["ci"]

    # onset
    o = ex["onset"]
    n["onset.calm"], n["onset.events"] = o["calm_months"], o["onsets"]
    for tag, key in [("only", "gprc_only"), ("ctrl", "controlled")]:
        v = o[key]
        n[f"onset.{tag}_or"] = v["or"]
        n[f"onset.{tag}_or_l"], n[f"onset.{tag}_or_u"] = v["or_ci"]
        n[f"onset.{tag}_p"] = v["p"]
        n[f"onset.{tag}_auc"] = v["auc"]

    # out-of-sample
    fc = tab(variant, "tableMS5_regime_forecast.csv").set_index(["regime", "model"])
    n["oos.all_r2"] = fc.at[("All", "M0_Persistence"), "R2_OOS"]
    n["oos.calm_r2"] = fc.at[("Calm", "M0_Persistence"), "R2_OOS"]
    n["oos.stress_r2"] = fc.at[("Stress", "M0_Persistence"), "R2_OOS"]
    cw = tab(variant, "tableMS5_cw_by_regime.csv").set_index(["regime", "challenger"])
    for rg in ["All", "Calm", "Stress"]:
        n[f"oos.cw_p_gprc.{rg}"] = cw.at[(rg, "M1_+GPRC"), "CW_p"]

    # inference robustness (ms7)
    ms7 = tab(variant, "tableMS7_inference_robust.csv").set_index("method")
    n["rob.dk_p"] = ms7.at["Driscoll-Kraay (NW4)", "p_value"]
    n["rob.webb_p"] = ms7.loc[[i for i in ms7.index if "Webb" in i][0], "p_value"]
    n["rob.rademacher_p"] = ms7.loc[[i for i in ms7.index if "Rademacher" in i][0], "p_value"]
    n["rob.timecluster_p"] = ms7.loc[[i for i in ms7.index if "Cluster by time" in i][0], "p_value"]
    n["rob.wild_time_p"] = ms7.loc[[i for i in ms7.index if "Wild Boot by time" in i][0], "p_value"]
    n["rob.block_boot_p"] = ms7.loc[[i for i in ms7.index if "Block Boot" in i][0], "p_value"]
    n["rob.country_cluster_p"] = ms7.loc[[i for i in ms7.index if "Cluster by country" in i][0], "p_value"]

    # eight regime definitions (ms8, time-clustered SE as submitted)
    ms8 = tab(variant, "tableMS8_regime_robust.csv")
    n["rob.reg8_negative"] = int((ms8["b3"] < 0).sum())
    n["rob.reg8_sig05_timecluster"] = int((ms8["p_value"] < 0.05).sum())
    n["rob.reg8_sig10_timecluster"] = int((ms8["p_value"] < 0.10).sum())
    for r in ms8.itertuples():
        n[f"reg8.{r.variant}.b3"] = r.b3
        n[f"reg8.{r.variant}.p_timecluster"] = r.p_value
        n[f"reg8.{r.variant}.n_stress"] = r.n_stress

    # placebos (ms9 + 5,000 permutations from cr3)
    ms9 = tab(variant, "tableMS9_placebo.csv").set_index("placebo")
    n["plac.return_p"] = ms9.at["Return placebo", "empirical_p"]
    for i in ms9.index:
        if i.startswith("Phase-shift"):
            n[f"plac.{i}.p"] = ms9.at[i, "empirical_p"]
    n["plac.perm_one_sided"] = ex["perm"]["p_one_sided"]
    n["plac.perm_two_sided"] = ex["perm"]["p_two_sided"]

    # filtered regime, nonlinear persistence
    f = ex["filtered"]
    n["filt.n_stress"], n["filt.agree"] = f["stress_months_filtered"], f["agreement_share"]
    n["filt.b3"], n["filt.p_hc3"], n["filt.p_dk4"] = f["binary"]["b3"], f["binary"]["hc3"]["p"], f["binary"]["dk4"]["p"]
    nl = ex["nonlin"]
    n["nonlin.sq.b3"], n["nonlin.sq.p"] = nl["plus_rv_sq"]["b3"], nl["plus_rv_sq"]["hc3"]["p"]
    n["nonlin.sqx.b3"], n["nonlin.sqx.p"] = (nl["plus_rv_sq_and_rv_x_stress"]["b3"],
                                             nl["plus_rv_sq_and_rv_x_stress"]["hc3"]["p"])
    n["nonlin.max_p"] = max(n["nonlin.sq.p"], n["nonlin.sqx.p"])

    # country interaction slopes (ms11)
    cs = tab(variant, "tableMS11_country_slopes.csv").set_index("country")
    for c in CC5:
        n[f"cslope.{c}.b"], n[f"cslope.{c}.p"] = cs.at[c, "gpr_x_stress"], cs.at[c, "p_value"]
    return {k: (float(v) if not isinstance(v, (list, str)) else v) for k, v in n.items()}


def checksum(raw: dict) -> str:
    raw = dict(raw)
    raw["rob.reg8_sig05_claimed"] = raw["rob.reg8_sig05_timecluster"]
    lines = ["# Checksum: numbers reproduced on the raw panel", "",
             "| Key | Paper | Reproduced | Match |", "|---|---:|---:|:---:|"]
    bad = 0
    for k, (v, dec) in PAPER.items():
        if k == "fx.b3_corrected":
            continue
        r = raw.get(k)
        ok = r is not None and round(abs(r - v), dec) <= 0.5 * 10 ** (-dec) + 1e-12
        if r is not None and not ok and abs(round(r, dec) - v) < 10 ** (-dec) * 1.01:
            ok = True   # rounding-boundary case
        bad += (not ok)
        lines.append(f"| `{k}` | {v} | {'' if r is None else round(r, dec + 1)} | {'✓' if ok else '✗'} |")
    lines += ["", f"**Mismatches: {bad} / {len(PAPER) - 1}**"]
    return "\n".join(lines)


def diff(raw: dict, cr: dict) -> str:
    lines = ["# Changes on the corrected-FX panel (raw → cr)", "",
             "Only keys that appear in the paper and change at the displayed precision are listed.", "",
             "| Khoá | PDF | raw | cr |", "|---|---:|---:|---:|"]
    for k, (v, dec) in PAPER.items():
        if k in ("fx.b3_corrected", "rob.reg8_sig05_claimed"):
            continue
        a, b = raw.get(k), cr.get(k)
        if a is None or b is None:
            continue
        if round(a, dec) != round(b, dec):
            lines.append(f"| `{k}` | {v} | {round(a, dec + 1)} | {round(b, dec + 1)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["raw", "cr"])
    v = ap.parse_args().variant
    nums = collect(v)
    save_json(nums, RES / f"numbers_{v}.json")
    if v == "raw":
        (RES / "checksum_raw.md").write_text(checksum(nums), encoding="utf-8")
        print(checksum(nums)[-400:])
    else:
        raw = load_json(RES / "numbers_raw.json")
        (RES / "diff_raw_vs_cr.md").write_text(diff(raw, nums), encoding="utf-8")
        print(diff(raw, nums))
