"""A1b Monte Carlo stress designs (addendum of 27/10/2026: docs/analysis_plan/addendum_2026-10-27.md).

Base design: cell 3 of a1_montecarlo.py (variant c, delta-hat, beta_3 = 0) with its seeds
(20261013 + 30000 + r), panel construction, estimated regime and OLS inference settings.
  base  cell 3 itself (validation: must reproduce the base-flag p-values of the main run)
  H     heavy-tailed GPRC: the GPRC innovations h_t and u_it are Student-t with nu degrees of
        freedom, scaled to unit variance (u_it also by sigma_u). nu is chosen by `calibrate_nu`
        before any rejection rate is computed: the value in NU_GRID whose median share of the
        top 5 months in the sum of squares of the beta_3 regressor (reps 0-199) is closest to
        the observed share.
  S     serially correlated common volatility shock: f_t = 0.5 f_t-1 + sqrt(0.75) eps_t, where
        eps_t is the normal draw of cell 3 (what-if design, rho fixed).

Subcommands (resumable; every output is cached):
  check         the copy of the simulator reproduces a1_montecarlo.simulate_core for cell 3
  smoke         2 replications per design in a temporary folder; the output is discarded
  calibrate_nu  choice of nu (results/mc/stress/H_nu_calibration.json)
  run           1,000 replications of base, H and S in shards of 50 (results/mc/stress/<design>/)
  summarize     results/qf_a1b.json and results/tables/mc_stress.csv
Usage: python stage2/analysis/a1b_mc_stress.py <subcommand> [--workers 14]
"""
from __future__ import annotations

import os

for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1_montecarlo as a1  # noqa: E402
import qflib as q  # noqa: E402

OUT = q.RES / "mc" / "stress"
CELL = 3
NU_GRID = [2.5, 3.0, 4.0, 5.0, 8.0]
RHO_S = 0.5
REPS, SHARD, CALIB_REPS = 1000, 50, 200
DESIGNS = ["base", "H", "S"]
PROCEDURES = ["dk4", "dk6", "dk12", "dk4_fixedb", "hc3", "cl_month", "wcb_rademacher_enum", "wcb_webb",
              "wb_month", "mbb"]
MAIN_PROCEDURES = ["dk4", "dk4_fixedb", "hc3", "cl_month", "wcb_webb", "wb_month", "mbb"]   # addendum list


# ============================================================================= simulation
def simulate(par: dict, design: str, rng: np.random.RandomState, nu: float | None = None):
    """Copy of a1_montecarlo.simulate_core for cell 3 (delta_c = delta_s = delta-hat, beta_3 = 0),
    with the change of design H or S. The draws are taken in the same order, so the regime path and
    the volatility innovations equal those of cell 3 in H, and every draw equals cell 3 in S."""
    T, n_c = a1.BURN + a1.T_OBS, a1.N_C
    dc = ds = par["dhat"]
    s = np.zeros(T, dtype=int)
    s[0] = int(rng.rand() < (1 - par["p00"]) / (2 - par["p00"] - par["p11"]))
    ur = rng.rand(T)
    for t in range(1, T):
        stay = par["p11"] if s[t - 1] == 1 else par["p00"]
        s[t] = s[t - 1] if ur[t] < stay else 1 - s[t - 1]
    f = rng.randn(T)
    if design == "S":
        eps = f.copy()
        for t in range(1, T):
            f[t] = RHO_S * f[t - 1] + np.sqrt(1 - RHO_S ** 2) * eps[t]
    e = rng.randn(T, n_c) * par["sigma_e"]
    if design == "H":
        k = np.sqrt((nu - 2) / nu)
        h = rng.standard_t(nu, size=T) * k
        u = rng.standard_t(nu, size=(T, n_c)) * k * par["sigma_u"]
    else:
        h = rng.randn(T)
        u = rng.randn(T, n_c) * par["sigma_u"]
    c = np.zeros((T, n_c))
    g = np.zeros((T, n_c))
    y = np.zeros((T, n_c))
    sc = par.get("sd_y", 1.0)
    for t in range(1, T):
        rho = par["rho_s"] if s[t] == 1 else par["rho_c"]
        dl = ds if s[t] == 1 else dc
        g[t] = par["phi_g"] * g[t - 1] + par["a"] * h[t] + u[t] + dl * e[t] / par["sigma_e"]
        c[t] = rho * c[t - 1] + par["lam"] * f[t] + e[t] + sc * (par["beta1"] * g[t - 1])
        y[t] = par["kappa"] * s[t] + c[t]
    sl = slice(a1.BURN, T)
    return y[sl], g[sl], s[sl], f[sl], e[sl]


def panel(design: str, rep: int, cal: dict, nu: float | None):
    """Simulated panel with the estimated regime in q.STRESS (as in a1_montecarlo.gr_one); None if
    the Markov-switching fit fails."""
    seed = a1.BASE_SEED + 10_000 * CELL + rep
    rng = np.random.RandomState(seed)
    y, g, s, f, e = simulate(cal["c"], design, rng, nu)
    p, _ = a1.build_panel(y, g, s, f, e, rng)
    fm = q.fit_ms(q.regional_mean(p), starts=3)
    if not fm["ok"]:
        return None, seed
    p[q.STRESS] = p.index.map(q.flag_threshold(fm["smoothed"]).to_dict())
    return p, seed


def ac1(z) -> float:
    z = np.asarray(z, float)
    return float(np.corrcoef(z[1:], z[:-1])[0, 1])


def leverage_stats(dz: dict) -> dict:
    """Tail and dependence diagnostics (same definitions as extra/diag_round3_premise.py)."""
    X, y, j, months = dz["X"], dz["y"], dz["pos"], dz["months"]
    _, resid, _ = q.ols_fit(X, y)
    others = np.delete(X, j, axis=1)
    xt = X[:, j] - others @ np.linalg.lstsq(others, X[:, j], rcond=None)[0]
    df = pd.DataFrame({"m": months, "u": resid, "s": xt * resid, "x2": xt ** 2})
    lev = df.groupby("m")["x2"].sum().sort_values(ascending=False)
    return {"top5_share": float(lev.head(5).sum() / lev.sum()),
            "kurt_gprc": float(pd.Series(dz["frame"]["gpr_country_z"].values).kurt()),
            "ac1_score": ac1(df.groupby("m")["s"].sum().sort_index()),
            "ac1_mean_resid": ac1(df.groupby("m")["u"].mean().sort_index())}


# ============================================================================= replications
def one_rep(design: str, rep: int, cal: dict, nu: float | None) -> dict:
    p, seed = panel(design, rep, cal, nu)
    out = {"design": design, "rep": rep, "seed": seed, "nu": nu if nu is not None else np.nan, "ok": p is not None}
    if p is None:
        return out
    dz = q.design(p, q.STRESS)
    o = q.ols_inference(dz, full=True, seed=seed)
    out["b3"] = o["b3"]
    for key in PROCEDURES:
        out[f"{key}_p"] = o[key]["p"]
        if "se" in o[key]:
            out[f"{key}_se"] = o[key]["se"]
    out["se_block1"] = q.mbb_ols(dz["X"], dz["y"], dz["months"], dz["pos"], B=399, block=1, seed=seed)[0]
    out.update(leverage_stats(dz))
    return out


def calib_rep(rep: int, cal: dict, nu: float) -> dict:
    """Calibration of nu: only the top-5 share and the GPRC kurtosis (no inference)."""
    p, _ = panel("H", rep, cal, nu)
    if p is None:
        return {"rep": rep, "nu": nu, "ok": False}
    st = leverage_stats(q.design(p, q.STRESS))
    return {"rep": rep, "nu": nu, "ok": True, "top5_share": st["top5_share"], "kurt_gprc": st["kurt_gprc"]}


def shard_path(design: str, k: int, base: Path = OUT) -> Path:
    return base / design / f"shard_{k:03d}.parquet"


def run_shard(design: str, k: int, cal: dict, nu: float | None, base: Path = OUT, per: int = SHARD) -> str:
    path = shard_path(design, k, base)
    if path.exists():
        return f"{design}/shard{k}: cached"
    t0 = time.time()
    rows = [one_rep(design, r, cal, nu) for r in range(k * per, (k + 1) * per)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, path)
    return f"{design}/shard{k}: {time.time() - t0:.0f}s"


def calib_shard(nu: float, k: int, cal: dict) -> str:
    path = OUT / "H_calib" / f"nu{nu:g}_shard_{k:03d}.parquet"
    if path.exists():
        return f"calib nu={nu:g}/shard{k}: cached"
    rows = [calib_rep(r, cal, nu) for r in range(k * SHARD, (k + 1) * SHARD)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, path)
    return f"calib nu={nu:g}/shard{k}: done"


# ============================================================================= subcommands
def check() -> bool:
    cal = a1.calibrate()
    par = cal["c"]
    same = True
    for rep in range(5):
        seed = a1.BASE_SEED + 10_000 * CELL + rep
        ref = a1.simulate_core(par, a1.T_OBS, a1.N_C, np.random.RandomState(seed), par["dhat"], par["dhat"], 0.0)
        new = simulate(par, "base", np.random.RandomState(seed))
        same &= all(np.array_equal(r_, n_) for r_, n_ in zip(ref, new))
        # H keeps the regime path and the volatility innovations of cell 3; S keeps the regime path
        h = simulate(par, "H", np.random.RandomState(seed), 4.0)
        s_ = simulate(par, "S", np.random.RandomState(seed))
        same &= np.array_equal(h[2], ref[2]) and np.array_equal(h[4], ref[4]) and np.array_equal(s_[2], ref[2])
    print("simulator copy reproduces cell 3 and keeps the stated draws:", same)
    return same


def smoke() -> None:
    cal = a1.calibrate()
    with tempfile.TemporaryDirectory() as d:
        for design, nu in [("base", None), ("H", 3.0), ("S", None)]:
            msg = run_shard(design, 0, cal, nu, base=Path(d), per=2)
            n = len(pd.read_parquet(shard_path(design, 0, Path(d))))
            print(f"smoke {msg}: {n} rows written (output discarded)")
        shutil.rmtree(d, ignore_errors=True)


def observed_top5() -> float:
    return leverage_stats(q.design(q.load_panel(), q.STRESS))["top5_share"]


def calibrate_nu(workers: int) -> dict:
    path = OUT / "H_nu_calibration.json"
    if path.exists():
        res = q.load_json(path)
        print("nu calibration: cached", res["chosen_nu"])
        return res
    cal = a1.calibrate()
    jobs = [(nu, k, cal) for nu in NU_GRID for k in range(CALIB_REPS // SHARD)]
    a1.pool_map(calib_shard, jobs, workers)
    df = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "H_calib").glob("*.parquet"))])
    df = df[df["ok"].astype(bool)]
    obs = observed_top5()
    grid = {f"{nu:g}": {"n": int((df["nu"] == nu).sum()),
                        "median_top5_share": float(df.loc[df["nu"] == nu, "top5_share"].median()),
                        "median_kurt_gprc": float(df.loc[df["nu"] == nu, "kurt_gprc"].median())} for nu in NU_GRID}
    chosen = min(NU_GRID, key=lambda nu: abs(grid[f"{nu:g}"]["median_top5_share"] - obs))
    res = {"observed_top5_share": obs, "grid": grid, "chosen_nu": chosen,
           "rule": "nu in NU_GRID with median top-5 share (reps 0-199) closest to the observed share"}
    q.save_json(res, path)
    print(res)
    return res


def run(workers: int) -> None:
    cal = a1.calibrate()
    nu = calibrate_nu(workers)["chosen_nu"]
    jobs = [(d, k, cal, nu if d == "H" else None) for d in DESIGNS for k in range(REPS // SHARD)
            if not shard_path(d, k).exists()]
    print(f"{len(jobs)} shards to compute (nu = {nu:g} for H)", flush=True)
    t0 = time.time()
    a1.pool_map(run_shard, jobs, workers)
    print(f"done in {time.time() - t0:.0f}s")


def summarize() -> dict:
    res = {"addendum": "docs/analysis_plan/addendum_2026-10-27.md", "cell": CELL, "rho_S": RHO_S,
           "nu_calibration": q.load_json(OUT / "H_nu_calibration.json"), "designs": {}}
    main = pd.concat([pd.read_parquet(f) for f in sorted((q.RES / "mc" / f"cell{CELL}").glob("shard_*.parquet"))])
    main = main.set_index("rep")
    rows = []
    for d in DESIGNS:
        parts = sorted((OUT / d).glob("shard_*.parquet"))
        if not parts:
            continue
        df_all = pd.concat([pd.read_parquet(f) for f in parts])
        df = df_all[df_all["ok"].astype(bool)].set_index("rep")
        out = {"n_total": int(len(df_all)), "n": int(len(df)), "n_ms_failed": int(len(df_all) - len(df)),
               "rejection_5pct": {m: a1.rate_ci(df[f"{m}_p"] < 0.05) for m in PROCEDURES},
               "mean_b3": float(df["b3"].mean()), "sd_b3": float(df["b3"].std()),
               "share_b3_le_-0.171": float((df["b3"] <= -0.171).mean()),
               "diagnostics_median": {k: float(df[k].median()) for k in
                                      ["top5_share", "kurt_gprc", "ac1_score", "ac1_mean_resid"]}}
        out["diagnostics_median"]["ratio_block1_cluster"] = float((df["se_block1"] / df["cl_month_se"]).median())
        out["diagnostics_median"]["ratio_block6_cluster"] = float((df["mbb_se"] / df["cl_month_se"]).median())
        if d == "base":   # validation against the main Monte Carlo run (same seeds, same pipeline)
            common = df.index.intersection(main.index)
            out["validation_vs_main_run"] = {
                "reps_compared": int(len(common)),
                "identical_p_values": {m: bool(np.allclose(df.loc[common, f"{m}_p"], main.loc[common, f"ols_base_{m}_p"],
                                                           rtol=0, atol=1e-12, equal_nan=True))
                                       for m in PROCEDURES if f"ols_base_{m}_p" in main},
                "identical_b3": bool(np.allclose(df.loc[common, "b3"], main.loc[common, "ols_base_b3"], rtol=0, atol=1e-12))}
        res["designs"][d] = out
        for m in MAIN_PROCEDURES:
            r = out["rejection_5pct"][m]
            rows.append({"design": d, "procedure": m, "n": r["n"], "rate": r["rate"], "mc_se": r["mc_se"],
                         "wilson_lo": r["wilson95"][0], "wilson_hi": r["wilson95"][1]})
    q.save_json(res, q.RES / "qf_a1b.json")
    (q.RES / "tables").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(q.RES / "tables" / "mc_stress.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "smoke", "calibrate_nu", "run", "summarize"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    elif a.cmd == "smoke":
        smoke()
    elif a.cmd == "calibrate_nu":
        calibrate_nu(a.workers)
    elif a.cmd == "run":
        run(a.workers)
    else:
        r = summarize()
        for d, v in r["designs"].items():
            print(d, {m: round(v["rejection_5pct"][m]["rate"], 3) for m in MAIN_PROCEDURES})
