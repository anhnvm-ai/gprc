"""A1 Monte Carlo (prespec P3-P5).

Question 1 (size): in a design like the data (5 markets x 196 months, common shocks,
persistence, an ESTIMATED Markov-switching regime), which inference procedure for beta_3
rejects a true H0: beta_3 = 0 about 5% of the time?
Question 2 (mechanical bias): does the paper's pipeline (regime from the cross-country
average log RV + a linear persistence control) produce a negative beta_3 when the true
beta_3 is zero, e.g. because GPRC reacts to volatility shocks and volatility mean-reverts
faster in stress (an alternative explanation)?

Subcommands (all resumable; every output is cached):
  calibrate          data moments -> DGP parameters (results/mc/calibration.json)
  run [--pilot]      simulate + estimate; shards of 50 replications (results/mc[_pilot]/cell*/)
  gr [--gr-reps N]   MS-re-estimating bootstrap on cell 3, reps 0 to N-1 (results/mc/gr/); N = 200 in the
                     analysis plan, 1000 in the addendum of 27/10/2026
  gr_check           recompute reps 0-9 of the MS-re-estimating bootstrap; they must equal shard 0
  summarize          tables, decision rules P3/P5, figure (results/qf_a1.json, results/tables/)
"""
from __future__ import annotations

import os

for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# make qflib (and, through it, the project's src/) importable in joblib worker processes
_HERE = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = os.pathsep.join([_HERE, str(Path(_HERE).parents[1]),
                                            os.environ.get("PYTHONPATH", "")])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qflib as q  # noqa: E402

MC = q.RES / "mc"
BASE_SEED = 20261013
T_OBS, BURN, N_C = 196, 120, 5
VNM_MISSING = 29                     # Vietnam enters in June 2012 (29 months after Jan 2010)
DATES = pd.date_range("2010-01-01", periods=T_OBS, freq="MS")
SHARD = 50

CELLS = {
    0: {"name": "a_H0", "var": "a", "delta": "0", "beta3": 0.0},
    1: {"name": "b_dhat_H0", "var": "a", "delta": "dhat", "beta3": 0.0},
    2: {"name": "b_3dhat_H0", "var": "a", "delta": "3dhat", "beta3": 0.0},
    3: {"name": "c_dhat_H0", "var": "c", "delta": "dhat", "beta3": 0.0},
    4: {"name": "c_3dhat_H0", "var": "c", "delta": "3dhat", "beta3": 0.0},
    5: {"name": "c_dstress_H0", "var": "c", "delta": "dstress", "beta3": 0.0},
    6: {"name": "a_H1_010", "var": "a", "delta": "0", "beta3": -0.10},
    7: {"name": "a_H1_017", "var": "a", "delta": "0", "beta3": -0.17},
    8: {"name": "c_dhat_H1_017", "var": "c", "delta": "dhat", "beta3": -0.17},
}
H0_CELLS = [0, 1, 2, 3, 4, 5]
CANDIDATES = ["dk4", "dk4_fixedb", "wcb_webb", "wb_month", "mbb"]          # prespec P3 order


# ============================================================================= calibration
def data_moments(p: pd.DataFrame) -> dict:
    """Moments of the real panel used to calibrate the DGP (z-units, regime = stored ex-post)."""
    q_ = p.dropna(subset=["log_rv_t_z", "gpr_country_z"]).copy()
    q_["y_l1"] = q_.groupby("country")["log_rv_t_z"].shift(1)
    q_["g_l1"] = q_.groupby("country")["gpr_country_z"].shift(1)
    qq = q_.dropna(subset=["y_l1", "g_l1"])
    s = qq[q.STRESS]
    ar = {k: float(np.polyfit(qq.loc[s == k, "y_l1"], qq.loc[s == k, "log_rv_t_z"], 1)[0]) for k in (0, 1)}
    rho_pool = float(np.polyfit(qq["y_l1"], qq["log_rv_t_z"], 1)[0])
    innov = qq["log_rv_t_z"] - rho_pool * qq["y_l1"]
    wide_y = q_.pivot_table(index=q_.index, columns="country", values="log_rv_t_z")
    wide_g = q_.pivot_table(index=q_.index, columns="country", values="gpr_country_z")
    iu = np.triu_indices(N_C, 1)
    st = p.groupby(p.index)[q.STRESS].first().dropna()
    n11 = ((st.shift(1) == 1) & (st == 1)).sum()
    n00 = ((st.shift(1) == 0) & (st == 0)).sum()
    base = q.ols_inference(q.design(p, q.STRESS), full=False)
    beta, _, _ = q.ols_fit(*[q.design(p, q.STRESS)[k] for k in ("X", "y")])
    names = q.design(p, q.STRESS)["names"]
    return {
        "p00": float(n00 / (st.shift(1) == 0).sum()), "p11": float(n11 / (st.shift(1) == 1).sum()),
        "share": float(st.mean()),
        "gap": float(q_.loc[q_[q.STRESS] == 1, "log_rv_t_z"].mean() - q_.loc[q_[q.STRESS] == 0, "log_rv_t_z"].mean()),
        "ar_calm": ar[0], "ar_stress": ar[1], "ar_pooled": rho_pool,
        "xcorr_y": float(np.nanmean(wide_y.corr().values[iu])),
        "xcorr_g": float(np.nanmean(wide_g.corr().values[iu])),
        "phi_g": float(np.polyfit(qq["g_l1"], qq["gpr_country_z"], 1)[0]),
        "corr_g_innov": float(np.corrcoef(qq["gpr_country_z"], innov)[0, 1]),
        "corr_g_innov_stress": float(np.corrcoef(qq.loc[s == 1, "gpr_country_z"], innov[s == 1])[0, 1]),
        "beta1": float(beta[names.index("gpr_country_z")]), "b3_data": base["b3"],
    }


def simulate_core(par: dict, n_months: int, n_c: int, rng: np.random.RandomState,
                  delta_c: float, delta_s: float, beta3: float):
    """Latent DGP. Regime s_t: Markov(p00, p11). Common shock f_t, idiosyncratic e_it.
      g_it = phi g_i,t-1 + a h_t + u_it + delta(s_t) e_it / sigma_e                (GPRC)
      c_it = rho(s_t) c_i,t-1 + lam f_t + e_it + sd_y (beta1 g_i,t-1 + beta3 g_i,t-1 s_t-1)
      y_it = kappa s_t + c_it                                                     (log RV)
    GPRC shifts next-month volatility and the shift decays at the volatility persistence
    rate, so Eq. (3) is correctly specified when rho does not depend on the regime
    (variant a) and misspecified when it does (variant c, the mechanism under study).
    beta1 and beta3 are in the paper's standardised units; sd_y (long-run SD of y, set by
    calibrate) converts them to the scale of y. GPRC has (about) unit variance.
    Returns arrays y, g, s, f, e (length n_months after BURN)."""
    T = BURN + n_months
    s = np.zeros(T, dtype=int)
    s[0] = int(rng.rand() < (1 - par["p00"]) / (2 - par["p00"] - par["p11"]))
    ur = rng.rand(T)
    for t in range(1, T):
        stay = par["p11"] if s[t - 1] == 1 else par["p00"]
        s[t] = s[t - 1] if ur[t] < stay else 1 - s[t - 1]
    f = rng.randn(T)
    e = rng.randn(T, n_c) * par["sigma_e"]
    h = rng.randn(T)
    u = rng.randn(T, n_c) * par["sigma_u"]
    c = np.zeros((T, n_c))
    g = np.zeros((T, n_c))
    y = np.zeros((T, n_c))
    sc = par.get("sd_y", 1.0)
    for t in range(1, T):
        rho = par["rho_s"] if s[t] == 1 else par["rho_c"]
        dl = delta_s if s[t] == 1 else delta_c
        g[t] = par["phi_g"] * g[t - 1] + par["a"] * h[t] + u[t] + dl * e[t] / par["sigma_e"]
        c[t] = (rho * c[t - 1] + par["lam"] * f[t] + e[t]
                + sc * (par["beta1"] * g[t - 1] + beta3 * g[t - 1] * s[t - 1]))
        y[t] = par["kappa"] * s[t] + c[t]
    sl = slice(BURN, T)
    return y[sl], g[sl], s[sl], f[sl], e[sl]


def sim_moments(y, g, s) -> dict:
    """Same statistics as data_moments, computed on a simulated balanced panel (z-scored)."""
    yz = (y - y.mean(0)) / y.std(0)
    gz = (g - g.mean(0)) / g.std(0)
    y0, y1, s1 = yz[:-1].ravel(), yz[1:].ravel(), np.repeat(s[1:], yz.shape[1])
    ar = {k: float(np.polyfit(y0[s1 == k], y1[s1 == k], 1)[0]) for k in (0, 1)}
    rho_pool = float(np.polyfit(y0, y1, 1)[0])
    innov = y1 - rho_pool * y0
    g1 = gz[1:].ravel()
    iu = np.triu_indices(yz.shape[1], 1)
    return {"gap": float(yz[s == 1].mean() - yz[s == 0].mean()), "ar_calm": ar[0], "ar_stress": ar[1],
            "ar_pooled": rho_pool, "xcorr_y": float(np.corrcoef(yz.T)[iu].mean()),
            "xcorr_g": float(np.corrcoef(gz.T)[iu].mean()),
            "corr_g_innov": float(np.corrcoef(g1, innov)[0, 1]),
            "corr_g_innov_stress": float(np.corrcoef(g1[s1 == 1], innov[s1 == 1])[0, 1])}


def calibrate() -> dict:
    from scipy.optimize import brentq, least_squares
    path = MC / "calibration.json"
    if path.exists():
        print("calibration: cached")
        return q.load_json(path)
    p = q.load_panel()
    m = data_moments(p)
    print("data moments:", {k: round(v, 4) for k, v in m.items()})
    n_long = 20_000
    # GPRC block: Var(g) ~ 1 and cross-correlation xcorr_g (delta = 0)
    phi = m["phi_g"]
    a = np.sqrt(m["xcorr_g"] * (1 - phi ** 2))
    sigma_u = np.sqrt((1 - m["xcorr_g"]) * (1 - phi ** 2))
    common = {"p00": m["p00"], "p11": m["p11"], "phi_g": phi, "a": float(a), "sigma_u": float(sigma_u),
              "beta1": m["beta1"]}

    def solve(variant: str) -> dict:
        targets = ["gap", "xcorr_y", "ar_calm"] + (["ar_stress"] if variant == "c" else [])

        def unpack(x):
            par = dict(common, kappa=x[0], lam=x[1], sigma_e=1.0, rho_c=x[2])
            par["rho_s"] = x[3] if variant == "c" else x[2]
            return par

        def resid(x):
            y, g, s, _, _ = simulate_core(unpack(x), n_long, N_C, np.random.RandomState(7), 0.0, 0.0, 0.0)
            mm = sim_moments(y, g, s)
            return [mm[t] - m[t] for t in targets]

        x0 = [1.0, 0.8, m["ar_calm"]] + ([m["ar_stress"]] if variant == "c" else [])
        lb = [0.0, 0.0, -0.5] + ([-0.5] if variant == "c" else [])
        ub = [5.0, 5.0, 0.95] + ([0.95] if variant == "c" else [])
        sol = least_squares(resid, x0, bounds=(lb, ub), diff_step=1e-2, max_nfev=200)
        par = unpack(sol.x)
        y, g, s, _, _ = simulate_core(par, n_long, N_C, np.random.RandomState(7), 0.0, 0.0, 0.0)
        par["fit"] = {"targets": targets, "sim": sim_moments(y, g, s), "cost": float(sol.cost)}
        par["sd_y"] = float(y.std(0).mean())          # converts standardised betas to the scale of y

        def gap_delta(dc, ds, key):
            y_, g_, s_, _, _ = simulate_core(par, n_long, N_C, np.random.RandomState(11), dc, ds, 0.0)
            return sim_moments(y_, g_, s_)[key]

        dhat = brentq(lambda d: gap_delta(d, d, "corr_g_innov") - m["corr_g_innov"], -1.0, 1.0, xtol=1e-4)
        # stress-specific reaction: delta in calm months fixed at dhat, delta in stress months
        # calibrated to the correlation observed in stress months
        try:
            dstress = brentq(lambda d: gap_delta(dhat, d, "corr_g_innov_stress") - m["corr_g_innov_stress"],
                             -2.0, 2.0, xtol=1e-4)
        except ValueError:
            dstress = float("nan")
        par["dstress_calibrated"] = bool(np.isfinite(dstress))
        par["dhat"], par["dstress"] = float(dhat), float(dstress if np.isfinite(dstress) else dhat)
        return par

    out = {"data": m, "a": solve("a"), "c": solve("c")}
    # fixed Lasso penalties for the Monte Carlo DML (LassoCV on the real data, prespec P4)
    from src.models.dml import _make_model
    d, ctrl = q.dml_frame(p, q.STRESS)
    d = d.dropna(subset=[q.OUTCOME, "GxS"] + ctrl)
    alphas = {}
    for key, col in [("alpha_y", q.OUTCOME), ("alpha_d", "GxS")]:
        mdl = _make_model("lasso").fit(d[ctrl].values.astype(float), d[col].values.astype(float))
        alphas[key] = float(mdl.named_steps["model"].alpha_)
    out["lasso"] = alphas
    q.save_json(out, path)
    for v in ("a", "c"):
        print(f"variant {v}: params", {k: round(val, 4) for k, val in out[v].items() if isinstance(val, float)})
        print(f"   simulated moments", {k: round(val, 4) for k, val in out[v]["fit"]["sim"].items()})
    print("lasso alphas", alphas)
    return out


# ============================================================================= one replication
def build_panel(y, g, s, f, e, rng) -> tuple[pd.DataFrame, pd.Series]:
    """Simulated observed panel with the same columns / transformations as qflib.load_panel."""
    rows = []
    T = len(s)
    vix = 0.8 * f + 0.6 * rng.randn(T)
    sp = -0.5 * f + rng.randn(T)
    us = np.zeros(T)
    for t in range(1, T):
        us[t] = 0.95 * us[t - 1] + 0.3 * rng.randn()
    wti = rng.randn(T)
    fx = np.zeros((T, N_C))
    for t in range(1, T):
        fx[t] = 0.3 * fx[t - 1] + rng.randn(N_C)
    ret = -0.3 * e / e.std() + rng.randn(T, N_C)
    for i, c in enumerate(q.CC5):
        t0 = VNM_MISSING if c == "VNM" else 0
        rows.append(pd.DataFrame({
            "country": c, "log_rv": y[t0:, i], "gpr_country": g[t0:, i], "vix_m": vix[t0:],
            "dlog_fx_m": fx[t0:, i], "sp500_ret_m": sp[t0:], "us10y_m": us[t0:], "wti_ret_m": wti[t0:],
            "ret_m": ret[t0:, i]}, index=DATES[t0:]))
    p = pd.concat(rows).sort_index(kind="stable")
    p.index.name = "date"
    p["log_rv_lead1"] = p.groupby("country")["log_rv"].shift(-1)
    p["ret_lead1"] = p.groupby("country")["ret_m"].shift(-1)
    p["log_rv_t"] = p["log_rv"]
    for col in q.ZCOLS:
        p[col + "_z"] = q.zscore_within(p, col)
    return p, pd.Series(s.astype(float), index=DATES)


def one_rep(cell_id: int, rep: int, cal: dict, full: bool = True) -> dict:
    cell = CELLS[cell_id]
    par = cal[cell["var"]]
    dc = {"0": 0.0, "dhat": par["dhat"], "3dhat": 3 * par["dhat"], "dstress": par["dhat"]}[cell["delta"]]
    ds = {"0": 0.0, "dhat": par["dhat"], "3dhat": 3 * par["dhat"], "dstress": par["dstress"]}[cell["delta"]]
    seed = BASE_SEED + 10_000 * cell_id + rep
    rng = np.random.RandomState(seed)
    y, g, s, f, e = simulate_core(par, T_OBS, N_C, rng, dc, ds, cell["beta3"])
    p, s_true = build_panel(y, g, s, f, e, rng)
    out = {"cell": cell_id, "rep": rep, "seed": seed, "true_share": float(s_true.mean())}

    fm = q.fit_ms(q.regional_mean(p), starts=3)
    out["ms_ok"] = bool(fm["ok"])
    flags = {"oracle": s_true}
    if fm["ok"]:
        flags["base"] = q.flag_threshold(fm["smoothed"])
        flags["top"] = q.flag_top_share(fm["smoothed"])
        flags["filt"] = q.flag_threshold(fm["filtered"])
        out["base_months"] = int(flags["base"].sum())
        out["agree_base_oracle"] = float((flags["base"] == s_true.reindex(flags["base"].index)).mean())
    try:
        p["S_loco"] = q.loco_flags(p, starts=3, rule="top").values
    except Exception:
        p["S_loco"] = np.nan
    for name, fl in flags.items():
        p[f"S_{name}"] = p.index.map(fl.to_dict())

    for name in ["base", "oracle", "top", "filt", "loco"]:
        col = f"S_{name}"
        if col not in p or p[col].isna().all():
            continue
        is_full = full and name in ("base", "oracle")
        o = q.ols_inference(q.design(p, col), full=is_full, seed=seed)
        out[f"ols_{name}_b3"] = o["b3"]
        for key, val in o.items():
            if isinstance(val, dict):
                for kk, vv in val.items():
                    out[f"ols_{name}_{key}_{kk}"] = vv
    if full:
        mdl_y, mdl_d = q.lasso_fixed(cal["lasso"]["alpha_y"]), q.lasso_fixed(cal["lasso"]["alpha_d"])
        for name in ["base", "oracle"]:
            col = f"S_{name}"
            if col not in p:
                continue
            for sch in ["shuffled", "blocked"]:
                r = q.dml_interaction(p, col, scheme=sch, model_y=mdl_y, model_d=mdl_d, seed=seed,
                                      dk_lags=(4,))
                out[f"dml_{name}_{sch}_theta"] = r["theta"]
                out[f"dml_{name}_{sch}_iid_p"] = r["iid"]["p"]
                out[f"dml_{name}_{sch}_dk4_p"] = r["dk4"]["p"]
                out[f"dml_{name}_{sch}_dk4_se"] = r["dk4"]["se"]
    return out


def run_shard(cell_id: int, k: int, cal: dict, out_dir: Path, reps_per_shard: int) -> str:
    path = out_dir / f"cell{cell_id}" / f"shard_{k:03d}.parquet"
    if path.exists():
        return f"cell{cell_id}/shard{k}: cached"
    t0 = time.time()
    rows = [one_rep(cell_id, rep, cal) for rep in range(k * reps_per_shard, (k + 1) * reps_per_shard)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, path)
    return f"cell{cell_id}/shard{k}: {time.time() - t0:.0f}s"


def pool_map(fn, jobs: list[tuple], workers: int) -> None:
    """Run fn(*job) in worker processes. ProcessPoolExecutor waits on events instead of
    joblib's time.sleep polling loop, which failed with WinError 1450 under heavy load.
    A failed job is logged and skipped; rerunning the command recomputes only missing shards."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # No max_tasks_per_child: with it the executor hung once the first workers retired
    # (CPython issue; 8 workers x 4 tasks = 32 shards, then no progress). The memory growth it
    # guarded against was fixed at its source (qflib.mbb_ols: no datetime64 dict keys).
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *job): job for job in jobs}
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as err:                       # keep the other shards running
                print(f"FAILED {futs[fut]}: {err!r}", flush=True)


def run(pilot: bool, workers: int, cells: list[int]) -> None:
    cal = calibrate()
    q.fixedb_null(195, 4)                                   # build the cache once, before forking
    out_dir = q.RES / ("mc_pilot" if pilot else "mc")
    reps = 50 if pilot else 1000
    per = 25 if pilot else SHARD
    jobs = [(c, k, cal, out_dir, per) for c in cells for k in range(reps // per)
            if not (out_dir / f"cell{c}" / f"shard_{k:03d}.parquet").exists()]
    t0 = time.time()
    print(f"{len(jobs)} shards to compute", flush=True)
    pool_map(run_shard, jobs, workers)
    print(f"done in {time.time() - t0:.0f}s")


# ============================================================================= supplementary DML designs
SUPP_CELLS = [0, 3, 5, 7, 8]


def supp_rep(cell_id: int, rep: int, cal: dict) -> dict:
    """Same simulated data as one_rep (same seed). DML designs not in the main run:
    reverse cross-fitting (Ciganovic et al. 2026; fold-mean, and pooled slope - exploratory)
    and the interacted 'by regime' DML (qflib.dml_by_regime - exploratory). Lasso, fixed alpha."""
    cell = CELLS[cell_id]
    par = cal[cell["var"]]
    dc = {"0": 0.0, "dhat": par["dhat"], "3dhat": 3 * par["dhat"], "dstress": par["dhat"]}[cell["delta"]]
    ds = {"0": 0.0, "dhat": par["dhat"], "3dhat": 3 * par["dhat"], "dstress": par["dstress"]}[cell["delta"]]
    seed = BASE_SEED + 10_000 * cell_id + rep
    rng = np.random.RandomState(seed)
    y, g, s, f, e = simulate_core(par, T_OBS, N_C, rng, dc, ds, cell["beta3"])
    p, s_true = build_panel(y, g, s, f, e, rng)
    out = {"cell": cell_id, "rep": rep}
    fm = q.fit_ms(q.regional_mean(p), starts=3)
    flags = {"oracle": s_true}
    if fm["ok"]:
        flags["base"] = q.flag_threshold(fm["smoothed"])
    mdl_y, mdl_d = q.lasso_fixed(cal["lasso"]["alpha_y"]), q.lasso_fixed(cal["lasso"]["alpha_d"])
    cty = pd.get_dummies(p["country"], prefix="cty", drop_first=True).astype(float)
    for name, fl in flags.items():
        col = f"S_{name}"
        p[col] = p.index.map(fl.to_dict())
        for pool in ["fold_mean", "pooled"]:
            r = q.dml_interaction(p, col, scheme="reverse", model_y=mdl_y, model_d=mdl_d, seed=seed,
                                  dk_lags=(4,), pool=pool)
            out[f"rev_{pool}_{name}_theta"], out[f"rev_{pool}_{name}_dk4_p"] = r["theta"], r["dk4"]["p"]
            out[f"rev_{pool}_{name}_dk4_se"] = r["dk4"]["se"]
        d = pd.concat([p, cty], axis=1)
        ctrl = [col, "log_rv_t_z"] + q.CTRL + list(cty.columns)
        d = d.dropna(subset=[q.OUTCOME, "gpr_country_z"] + ctrl)
        for sch in ["shuffled", "blocked"]:
            r = q.dml_by_regime(d[q.OUTCOME].values, d["gpr_country_z"].values, d[col].values,
                                d[ctrl].values.astype(float), d.index.values, scheme=sch,
                                model_y=mdl_y, model_g=q.lasso_fixed(cal["lasso"]["alpha_d"]), seed=seed)
            out[f"byreg_{sch}_{name}_delta"], out[f"byreg_{sch}_{name}_dk4_p"] = r["delta"], r["dk4"]["p"]
            out[f"byreg_{sch}_{name}_dk4_se"] = r["dk4"]["se"]
    return out


def supp_shard(cell_id: int, k: int, cal: dict) -> str:
    path = MC / "supp" / f"cell{cell_id}" / f"shard_{k:03d}.parquet"
    if path.exists():
        return f"supp cell{cell_id}/shard{k}: cached"
    t0 = time.time()
    rows = [supp_rep(cell_id, r, cal) for r in range(k * SHARD, (k + 1) * SHARD)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, path)
    return f"supp cell{cell_id}/shard{k}: {time.time() - t0:.0f}s"


def supp(workers: int) -> None:
    cal = calibrate()
    jobs = [(c, k, cal) for c in SUPP_CELLS for k in range(1000 // SHARD)
            if not (MC / "supp" / f"cell{c}" / f"shard_{k:03d}.parquet").exists()]
    print(f"{len(jobs)} supplementary shards to compute", flush=True)
    pool_map(supp_shard, jobs, workers)


def summarize_supp() -> dict:
    parts = sorted((MC / "supp").glob("cell*/shard_*.parquet"))
    if not parts:
        return {}
    df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    res = {"note": "reverse fold-mean is the prespec design; reverse pooled and by-regime DML are exploratory"}
    for c, g in df.groupby("cell"):
        b0 = CELLS[c]["beta3"]
        d = {"n": int(len(g))}
        for est in ["rev_fold_mean", "rev_pooled"]:
            for fl in ["base", "oracle"]:
                th = g[f"{est}_{fl}_theta"]
                d[f"{est}_{fl}"] = {"mean": float(th.mean()), "median": float(th.median()),
                                    "sd": float(th.std()), "rej5": float((g[f"{est}_{fl}_dk4_p"] < 0.05).mean()),
                                    "coverage95": float(((th - b0).abs() <= 1.96 * g[f"{est}_{fl}_dk4_se"]).mean())}
        for sch in ["shuffled", "blocked"]:
            for fl in ["base", "oracle"]:
                dl = g[f"byreg_{sch}_{fl}_delta"]
                d[f"byreg_{sch}_{fl}"] = {"mean": float(dl.mean()), "sd": float(dl.std()),
                                          "rej5": float((g[f"byreg_{sch}_{fl}_dk4_p"] < 0.05).mean()),
                                          "coverage95": float(((dl - b0).abs() <= 1.96 * g[f"byreg_{sch}_{fl}_dk4_se"]).mean())}
        res[CELLS[c]["name"]] = d
    q.save_json(res, q.RES / "qf_a1_supp.json")
    return res


# ============================================================================= MS-re-estimating bootstrap
def gr_one(rep: int, cal: dict) -> dict:
    cell_id = 3
    cell, par = CELLS[cell_id], cal["c"]
    seed = BASE_SEED + 10_000 * cell_id + rep
    rng = np.random.RandomState(seed)
    y, g, s, f, e = simulate_core(par, T_OBS, N_C, rng, par["dhat"], par["dhat"], cell["beta3"])
    p, _ = build_panel(y, g, s, f, e, rng)
    fm = q.fit_ms(q.regional_mean(p), starts=3)
    if not fm["ok"]:
        return {"rep": rep, "ok": False}
    p[q.STRESS] = p.index.map(q.flag_threshold(fm["smoothed"]).to_dict())
    o = q.ols_inference(q.design(p, q.STRESS), full=False)
    gb = q.ms_regime_bootstrap(p, B=199, block=6, seed=seed, starts=3, rule="p05")
    return {"rep": rep, "ok": True, "b3": o["b3"], "dk4_p": o["dk4"]["p"], "dk4_se": o["dk4"]["se"],
            "gr_se": gb["se"], "gr_se_iqr": gb["se_iqr"], "gr_p": gb["p"], "gr_B": gb["B_used"],
            "gr_degenerate": gb["degenerate_skipped"]}


def gr_shard(k: int, cal: dict) -> str:
    out = MC / "gr"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{k:03d}.parquet"
    if path.exists():
        return f"gr shard {k}: cached"
    rows = [gr_one(r, cal) for r in range(k * 10, (k + 1) * 10)]
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, path)
    return f"gr shard {k}: done"


def gr(workers: int, reps: int = 200) -> None:
    cal = calibrate()
    pool_map(gr_shard, [(k, cal) for k in range(reps // 10) if not (MC / "gr" / f"shard_{k:03d}.parquet").exists()],
             workers)
    # the same bootstrap on the real data (B = 999)
    real = MC / "gr_real.json"
    if not real.exists():
        res = q.ms_regime_bootstrap(q.load_panel(), B=999, block=6, seed=42, starts=3, rule="p05")
        q.save_json({k: v for k, v in res.items() if k != "draws"}, real)
    print("real data:", q.load_json(real))


def gr_check() -> dict:
    """Addendum of 27/10/2026, R1: reps 0-9 recomputed with the current code must equal shard 0."""
    cal = calibrate()
    old = pd.read_parquet(MC / "gr" / "shard_000.parquet").reset_index(drop=True)
    new = pd.DataFrame([gr_one(r, cal) for r in range(10)])
    cols = [c for c in old.columns if c in new.columns]
    diff = {c: float(np.nanmax(np.abs(old[c].astype(float) - new[c].astype(float)))) for c in cols
            if not old[c].equals(new[c])}
    res = {"identical": not diff and len(cols) == len(old.columns) == len(new.columns), "columns": cols,
           "max_abs_diff_of_differing_columns": diff}
    q.save_json(res, MC / "gr_check.json")
    print(res)
    return res


# ============================================================================= summary
def rate_ci(reject) -> dict:
    """Rejection rate with its Monte Carlo standard error and 95% Wilson interval (addendum, 27/10/2026)."""
    x = np.asarray(reject, dtype=bool)
    n, k, z = len(x), int(x.sum()), 1.959964
    r = k / n
    centre = (r + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"n": n, "rejections": k, "rate": r, "mc_se": float(np.sqrt(r * (1 - r) / n)),
            "wilson95": [float(centre - half), float(centre + half)]}


def load_cells(out_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(out_dir.glob("cell*/shard_*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize(pilot: bool = False) -> dict:
    out_dir = q.RES / ("mc_pilot" if pilot else "mc")
    df = load_cells(out_dir)
    cal = q.load_json(MC / "calibration.json")
    res = {"n_reps": df.groupby("cell").size().to_dict(), "calibration": cal}
    methods = {"dk4": "ols_{f}_dk4_p", "dk6": "ols_{f}_dk6_p", "dk12": "ols_{f}_dk12_p",
               "dk4_fixedb": "ols_{f}_dk4_fixedb_p", "hc3": "ols_{f}_hc3_p", "cl_month": "ols_{f}_cl_month_p",
               "wcb_rademacher_enum": "ols_{f}_wcb_rademacher_enum_p", "wcb_webb": "ols_{f}_wcb_webb_p",
               "wb_month": "ols_{f}_wb_month_p", "mbb": "ols_{f}_mbb_p",
               "dml_shuffled_iid": "dml_{f}_shuffled_iid_p", "dml_shuffled_dk4": "dml_{f}_shuffled_dk4_p",
               "dml_blocked_iid": "dml_{f}_blocked_iid_p", "dml_blocked_dk4": "dml_{f}_blocked_dk4_p"}
    # rejection rates at 5% (size under H0 cells, power under H1 cells)
    rej = {}
    for flag in ["base", "oracle"]:
        rej[flag] = {}
        for c, g in df.groupby("cell"):
            rej[flag][CELLS[c]["name"]] = {m: float((g[col.format(f=flag)] < 0.05).mean())
                                           for m, col in methods.items() if col.format(f=flag) in g}
    res["rejection_5pct"] = rej
    # bias / RMSE of the point estimates
    est = {"ols_base": "ols_base_b3", "ols_oracle": "ols_oracle_b3", "ols_top": "ols_top_b3",
           "ols_filt": "ols_filt_b3", "ols_loco": "ols_loco_b3",
           "dml_base_shuffled": "dml_base_shuffled_theta", "dml_base_blocked": "dml_base_blocked_theta",
           "dml_oracle_shuffled": "dml_oracle_shuffled_theta", "dml_oracle_blocked": "dml_oracle_blocked_theta"}
    bias = {}
    for c, g in df.groupby("cell"):
        b0 = CELLS[c]["beta3"]
        bias[CELLS[c]["name"]] = {k: {"mean": float(g[col].mean()), "bias": float(g[col].mean() - b0),
                                      "rmse": float(np.sqrt(((g[col] - b0) ** 2).mean())),
                                      "sd": float(g[col].std())} for k, col in est.items() if col in g}
    res["estimates"] = bias
    # coverage of 95% DK intervals (normal and fixed-b critical values), baseline flag
    cv_fb = float(np.quantile(q.fixedb_null(195, 4), 0.95))
    cov = {}
    for c, g in df.groupby("cell"):
        b0 = CELLS[c]["beta3"]
        dev = (g["ols_base_b3"] - b0).abs()
        cov[CELLS[c]["name"]] = {"dk4_normal": float((dev <= 1.96 * g["ols_base_dk4_se"]).mean()),
                                 "dk4_fixedb": float((dev <= cv_fb * g["ols_base_dk4_se"]).mean()),
                                 "hc3": float((dev <= 1.96 * g["ols_base_hc3_se"]).mean())}
    res["coverage_95"] = cov
    res["fixedb_cv_95_T195_L4"] = cv_fb
    # ---- prespec P3: primary inference procedure (baseline flag, H0 cells 0-5)
    max_size = {m: max(rej["base"][CELLS[c]["name"]][m] for c in H0_CELLS if CELLS[c]["name"] in rej["base"])
                for m in CANDIDATES}
    ok = [m for m in CANDIDATES if max_size[m] <= 0.07]
    chosen = min(ok, key=lambda m: (abs(max_size[m] - 0.05), CANDIDATES.index(m))) if ok else \
        min(CANDIDATES, key=lambda m: max_size[m])
    res["P3"] = {"max_size_over_H0_cells": max_size, "eligible_le_7pct": ok, "chosen": chosen,
                 "rule": "closest to 5% among procedures with max size <= 7%; else smallest size"}
    # ---- prespec P5: mechanical bias and simulation-based p-values
    b_c3 = res["estimates"].get("c_dhat_H0", {}).get("ols_base", {}).get("mean", np.nan)
    sim_p = {}
    for c in [3, 5]:
        g = df[df["cell"] == c]
        if len(g):
            sim_p[CELLS[c]["name"]] = {"ols_base_le_-0.171": float((g["ols_base_b3"] <= -0.171).mean()),
                                       "dml_blocked_le_-0.202": float((g["dml_base_blocked_theta"] <= -0.202).mean())}
    res["P5"] = {"mean_b3_cell3": b_c3, "material": bool(abs(b_c3) >= 0.057), "threshold": 0.057,
                 "simulation_p": sim_p}
    gr_parts = sorted((MC / "gr").glob("shard_*.parquet")) if not pilot else []
    if gr_parts:
        gall = pd.concat([pd.read_parquet(f) for f in gr_parts])
        gdf = gall[gall["ok"].astype(bool)]
        res["gr_bootstrap_cell3"] = {"n": int(len(gdf)), "n_total": int(len(gall)),
                                     "n_ms_failed": int(len(gall) - len(gdf)),
                                     "size_5pct": float((gdf["gr_p"] < 0.05).mean()),
                                     "size_5pct_ci": rate_ci(gdf["gr_p"] < 0.05),
                                     "size_dk4_same_reps": float((gdf["dk4_p"] < 0.05).mean()),
                                     "size_dk4_same_reps_ci": rate_ci(gdf["dk4_p"] < 0.05),
                                     "mean_se_gr": float(gdf["gr_se"].mean()),
                                     "median_se_gr": float(gdf["gr_se"].median()),
                                     "mean_se_gr_iqr": float(gdf["gr_se_iqr"].mean()) if "gr_se_iqr" in gdf else None,
                                     "mean_degenerate_draws": float(gdf["gr_degenerate"].mean()) if "gr_degenerate" in gdf else None,
                                     "mean_se_dk4": float(gdf["dk4_se"].mean()),
                                     "sd_b3": float(gdf["b3"].std())}
        res["gr_bootstrap_real"] = q.load_json(MC / "gr_real.json")
    name = "qf_a1_pilot.json" if pilot else "qf_a1.json"
    q.save_json(res, q.RES / name)
    write_tables(res, pilot)
    if not pilot:
        figure(df)
    return res


def write_tables(res: dict, pilot: bool) -> None:
    tdir = q.RES / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for flag in ["base", "oracle"]:
        for cname, d in res["rejection_5pct"][flag].items():
            rows.append({"flag": flag, "cell": cname, **{k: round(v, 3) for k, v in d.items()}})
    pd.DataFrame(rows).to_csv(tdir / ("mc_rejection_pilot.csv" if pilot else "mc_rejection.csv"), index=False)
    rows = []
    for cname, d in res["estimates"].items():
        for est, v in d.items():
            rows.append({"cell": cname, "estimator": est, **{k: round(x, 4) for k, x in v.items()}})
    pd.DataFrame(rows).to_csv(tdir / ("mc_estimates_pilot.csv" if pilot else "mc_estimates.csv"), index=False)


def figure(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, c in zip(axes, [0, 3, 5]):
        g = df[df["cell"] == c]
        bins = np.linspace(-0.6, 0.6, 49)
        ax.hist(g["ols_oracle_b3"], bins=bins, histtype="step", color="0.5", lw=1.4, label="true regime")
        ax.hist(g["ols_base_b3"], bins=bins, histtype="step", color="k", lw=1.4, ls="--",
                label="estimated regime")
        ax.axvline(-0.171, color="k", lw=1)
        ax.set_title(CELLS[c]["name"], fontsize=9)
        ax.set_xlabel(r"$\hat\beta_3$ under $H_0$", fontsize=9)
    axes[0].set_ylabel("Frequency", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fdir = q.RES / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / "mc_null_distributions.pdf")
    fig.savefig(fdir / "mc_null_distributions.png", dpi=150)


# ============================================================================= main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "run", "gr", "gr_check", "summarize", "timing", "supp",
                                    "supp_summary"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--gr-reps", type=int, default=200)
    ap.add_argument("--cells", default="0-8")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.cells.split("-")) if "-" in a.cells else (int(a.cells), int(a.cells))
    if a.cmd == "calibrate":
        calibrate()
    elif a.cmd == "timing":
        cal = calibrate()
        t0 = time.time()
        r = one_rep(3, 0, cal)
        print(f"one replication: {time.time() - t0:.1f}s; keys {len(r)}; ols_base_b3 {r.get('ols_base_b3')}")
        # scale check: with the true regime and a long panel, beta3 = -0.17 (standardised)
        # must be recovered by Eq. (3) on z-scored data
        for var, dkey in [("a", None), ("c", None), ("c", "dhat")]:
            par = cal[var]
            d = par[dkey] if dkey else 0.0
            y, g, s, f, e = simulate_core(par, 20000, N_C, np.random.RandomState(3), d, d, -0.17)
            yz, gz = (y - y.mean(0)) / y.std(0), (g - g.mean(0)) / g.std(0)
            Y = yz[1:].ravel()
            S = np.repeat(s[:-1], N_C).astype(float)
            G = gz[:-1].ravel()
            X = np.column_stack([np.ones_like(Y), G, S, G * S, yz[:-1].ravel()])
            print(f"scale check, variant {var}, delta={d:.3f} (target -0.17):",
                  round(float(np.linalg.lstsq(X, Y, rcond=None)[0][3]), 4))
    elif a.cmd == "run":
        run(a.pilot, a.workers, list(range(lo, hi + 1)))
    elif a.cmd == "gr":
        gr(a.workers, a.gr_reps)
    elif a.cmd == "gr_check":
        gr_check()
    elif a.cmd == "supp":
        supp(a.workers)
    elif a.cmd == "supp_summary":
        print(summarize_supp())
    else:
        summarize(a.pilot)
