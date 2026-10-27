"""Reproduction tests: qflib must give the earlier-version numbers (corrected-FX panel)
before it is used for any new analysis (prespec, principle 3).

References (full precision):
  stage1/results/extra_cr.json   (Table 4, filtered regime)
  stage1/results/phaseB_cr.json  (DML with DK, LOCO, balanced sample)
  stage1/results/numbers_cr.json (bootstrap p-values of Table A2, 4 decimals)

Usage: python stage2/analysis/test_reproduce.py   -> writes results/test_reproduce.json
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402

REF = q.ROOT / "stage1" / "results"
TOL = 1e-6
checks = []


def check(name, got, want, tol=TOL):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    checks.append({"test": name, "got": got, "want": want, "ok": bool(ok)})
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:45s} got {got:+.6f}  want {want:+.6f}")


def main():
    t0 = time.time()
    extra = q.load_json(REF / "extra_cr.json")
    phb = q.load_json(REF / "phaseB_cr.json")
    num = q.load_json(REF / "numbers_cr.json")
    p = q.load_panel()

    print("T1 OLS Eq.(3) with persistence (Table 4 / Table A2)")
    dz = q.design(p, q.STRESS)
    o = q.ols_inference(dz, B_webb=999, B_month=999, B_mbb=499, seed=42)
    ref = extra["table4"]["persistence"]
    check("b3", o["b3"], ref["b3"])
    check("HC3 SE", o["hc3"]["se"], ref["hc3"]["se"])
    check("calendar-month cluster SE", o["cl_month"]["se"], ref["time_cluster"]["se"])
    for L in (4, 6, 12):
        check(f"DK{L} SE", o[f"dk{L}"]["se"], ref[f"dk{L}"]["se"])

    print("T2 OLS without persistence")
    o2 = q.ols_inference(q.design(p, q.STRESS, persistence=False), full=False)
    ref2 = extra["table4"]["no_persistence"]
    check("b3 (no persistence)", o2["b3"], ref2["b3"])
    check("DK4 p (no persistence)", o2["dk4"]["p"], ref2["dk4"]["p"])

    print("T3 Bootstraps (ms7: B=999 wild, B=499 block, seed 42; reference rounded to 4 dp)")
    X, y, j = dz["X"], dz["y"], dz["pos"]
    # Country-cluster wild bootstraps: the all-equal-weight patterns give |t*| = |t_obs| (ties).
    # ms7 counted ties only when the float comparison happened to succeed, so its p-value must
    # lie between the strict count and the count including ties.
    ties_info = {}
    for wname, key in [("rademacher", "rob.rademacher_p"), ("webb", "rob.webb_p")]:
        t_obs, t_star = q.wild_cluster_t(X, y, dz["country"], j, wname, 999, 42)
        a = np.abs(t_star)
        tie = np.abs(a - abs(t_obs)) <= q.TIE_RTOL * abs(t_obs)
        strict = a > abs(t_obs) * (1 + q.TIE_RTOL)
        lo, hi = strict.mean(), (strict | tie).mean()
        ref = num[key]
        ok = lo - 5e-4 <= ref <= hi + 5e-4
        checks.append({"test": f"wild cluster {wname}: ms7 p within [strict, with ties]",
                       "got": [float(lo), float(hi)], "want": ref, "ok": bool(ok)})
        print(f"  [{'OK ' if ok else 'FAIL'}] wild cluster {wname:10s} ms7 p {ref:.4f} in "
              f"[{lo:.4f}, {hi:.4f}] (ties: {int(tie.sum())} of 999 draws)")
        ties_info[wname] = {"ms7_p": ref, "strict": float(lo), "with_ties": float(hi),
                            "n_tie_draws": int(tie.sum())}
    ties_info["rademacher_enumerated_all_32_patterns"] = q.wild_cluster_p(
        X, y, dz["country"], j, enumerate_all=True)
    print(f"  -> exact Rademacher p over all 32 sign patterns (ties counted): "
          f"{ties_info['rademacher_enumerated_all_32_patterns']:.4f}")
    check("wild by month p", round(q.wild_cluster_p(X, y, dz["months"], j, "rademacher", 999, 42), 4),
          num["rob.wild_time_p"])
    se_bb, p_bb, _ = q.mbb_ols(X, y, dz["months"], j, B=499, block=6, seed=42)
    check("moving-block bootstrap p", round(p_bb, 4), num["rob.block_boot_p"])
    check("moving-block bootstrap SE (Table A2: 0.124)", round(se_bb, 3), 0.124)

    print("T4 Markov switching: smoothed regime = stored regime; filtered regime (Table 5)")
    reg = q.regional_mean(p)
    fm = q.fit_ms(reg, starts=10)
    stored = p.groupby(p.index)[q.STRESS].first().reindex(fm["smoothed"].index)
    check("smoothed P>0.5 equals stored regime (share of months)",
          float((q.flag_threshold(fm["smoothed"]).values == stored.values).mean()), 1.0)
    of = q.ols_inference(q.design(p, q.flag_threshold(fm["filtered"])), full=False)
    check("filtered regime b3", of["b3"], extra["filtered"]["binary"]["b3"])
    check("filtered regime DK4 p", of["dk4"]["p"], extra["filtered"]["binary"]["dk4"]["p"])

    print("T5 DML with month-aggregated DK (Table 4)")
    for sch in ["shuffled", "blocked"]:
        r = q.dml_interaction(p, q.STRESS, scheme=sch)
        ref = phb["dml_dk"][sch]
        check(f"DML {sch} theta", r["theta"], ref["theta"])
        check(f"DML {sch} iid SE", r["iid"]["se"], ref["iid"]["se"])
        check(f"DML {sch} DK4 SE", r["dk4"]["se"], ref["dk4"]["se"])

    print("T6 LOCO regime with baseline stress share (Table 5)")
    pl = p.copy()
    pl["stress_loco_top"] = q.loco_flags(p, starts=10, rule="top").values
    ol = q.ols_inference(q.design(pl, "stress_loco_top"), full=False)
    check("LOCO OLS b3", ol["b3"], phb["loco"]["ms_top"]["ols"]["b3"])
    check("LOCO OLS DK4 p", ol["dk4"]["p"], phb["loco"]["ms_top"]["ols"]["dk4"]["p"])
    rl = q.dml_interaction(pl, "stress_loco_top", scheme="blocked")
    check("LOCO time-aware DML theta", rl["theta"], phb["loco"]["ms_top"]["dml"]["blocked"]["theta"])
    check("LOCO time-aware DML DK4 p", rl["dk4"]["p"], phb["loco"]["ms_top"]["dml"]["blocked"]["dk4"]["p"])

    print("T7 Balanced five-market sample, baseline stress share (Table 5)")
    lr = p.pivot_table(index=p.index, columns="country", values="log_rv")
    months = lr.dropna().index
    regb = q.regional_mean(p, months=months)
    fb = q.fit_ms(regb, starts=10)
    pb = p[p.index.isin(months)].copy()
    pb["stress_bal"] = pb.index.map(q.flag_top_share(fb["smoothed"]).to_dict())
    ob = q.ols_inference(q.design(pb, "stress_bal"), full=False)
    check("balanced b3", ob["b3"], phb["balanced"]["ms_top"]["ols"]["b3"])
    check("balanced DK4 p", ob["dk4"]["p"], phb["balanced"]["ms_top"]["ols"]["dk4"]["p"])

    n_fail = sum(not c["ok"] for c in checks)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed in {time.time() - t0:.0f}s")
    q.save_json({"checks": checks, "passed": len(checks) - n_fail, "total": len(checks),
                 "wild_cluster_ties": ties_info,
                 "new_procedures_on_real_data": {"dk4_fixedb_p": o["dk4_fixedb"]["p"],
                                                 "wcb_rademacher_enumerated_p": o["wcb_rademacher_enum"]["p"],
                                                 "wcb_webb_B999_ties_counted_p": o["wcb_webb"]["p"]}},
                q.RES / "test_reproduce.json")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
