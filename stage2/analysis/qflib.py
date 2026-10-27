"""qflib: shared library for the Phase Q2 analyses of the paper.

The functions below re-implement, without changing the algorithms, the code that
produced the earlier-version numbers:
  * stage1/scripts/cr3_extra.py      (panel, OLS of Eq. 3, DK, Markov switching)
  * stage1/scripts/cr4_experiments.py (DML with month-aggregated DK, LOCO, balanced)
  * stage1/rerun/scripts/ms7_inference_robust.py (bootstraps)
  * src/models/dml.py                            (learners, blocked splits, cross-fitting)
`test_reproduce.py` checks that they give the earlier-version numbers.

New procedures (documented where defined): fixed-b p-values for Driscoll-Kraay,
vectorised wild bootstraps, additional cross-fitting schemes, and a moving-block
bootstrap that re-estimates the Markov-switching regime in every draw.
Pre-specification: ../prespec.md.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

QF = Path(__file__).resolve().parents[1]
ROOT = QF.parent
RES = QF / "results"
PANEL_PATH = ROOT / "stage1" / "rerun" / "data" / "processed" / "asean_panel_regimes.parquet"
sys.path.insert(0, str(ROOT))
from src.models.dml import _make_model, make_blocked_splits  # noqa: E402

CC5 = ["VNM", "THA", "IDN", "MYS", "PHL"]
STRESS = "stress_common"
OUTCOME = "log_rv_lead1_z"
CTRL = ["vix_m_z", "dlog_fx_m_z", "sp500_ret_m_z", "us10y_m_z", "wti_ret_m_z", "ret_m_z"]
ZCOLS = ["gpr_country", "vix_m", "dlog_fx_m", "log_rv_t", "log_rv_lead1", "ret_lead1",
         "sp500_ret_m", "us10y_m", "wti_ret_m", "ret_m"]
BASE_SHARE = 38 / 196
DK_LAGS = (4, 6, 12)
WEBB6 = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# =============================================================================== I/O
def to_jsonable(o):
    """Convert numpy / pandas objects (recursively) into JSON-serialisable values."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def save_json(obj, path: Path) -> None:
    """Atomic write: a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pval(b: float, se: float) -> float:
    """Two-sided normal p-value."""
    return float(2 * (1 - norm.cdf(abs(b / se))))


# =============================================================================== data
def load_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    """Corrected-FX panel, identical to cr3_extra.load_panel('cr'): log RV, its one-month
    lead and within-country z-scores (full-sample moments)."""
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    p = p.set_index("date").sort_index()
    p = p[p["country"].isin(CC5)].copy()
    p["log_rv"] = np.log(p["rv_m"].clip(lower=1e-8))
    p["log_rv_lead1"] = p.groupby("country")["log_rv"].shift(-1)
    p["ret_lead1"] = p.groupby("country")["ret_m"].shift(-1)
    p["log_rv_t"] = p["log_rv"]
    for c in ZCOLS:
        p[c + "_z"] = p.groupby("country")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    return p


def zscore_within(p: pd.DataFrame, col: str) -> pd.Series:
    return p.groupby("country")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))


def regional_mean(p: pd.DataFrame, exclude: str | None = None, months=None) -> pd.Series:
    """Cross-country average log RV by month (optionally without one market / on given months)."""
    q = p if exclude is None else p[p["country"] != exclude]
    r = q.groupby(q.index)["log_rv"].mean().dropna()
    return r if months is None else r[r.index.isin(months)]


# =============================================================================== Markov switching
def fit_ms(y, starts: int = 10, seed: int = 42) -> dict:
    """Two-state switching-mean, constant-variance Markov regression (same loop as
    cr3_extra.fit_ms): start 0 is the statsmodels default, starts 1.. perturb the
    default start values by N(0, 0.15) relative noise (RandomState(seed + start)).
    The stress state is the state whose smoothed-assigned months have the higher mean.
    Returns smoothed and filtered stress probabilities and the parameters."""
    if isinstance(y, pd.Series):
        y = y.dropna()
        idx, yv = y.index, np.asarray(y, dtype=float)
    else:
        yv = np.asarray(y, dtype=float)
        idx = pd.RangeIndex(len(yv))
    best = None
    for rep in range(starts):
        try:
            mod = MarkovRegression(yv, k_regimes=2, trend="c", switching_variance=False)
            if rep == 0:
                res = mod.fit(em_iter=40, maxiter=200)
            else:
                rng = np.random.RandomState(seed + rep)
                sp0 = mod.start_params * (1 + 0.15 * rng.randn(len(mod.start_params)))
                res = mod.fit(start_params=sp0, em_iter=40, maxiter=200)
            if np.isfinite(res.llf) and (best is None or res.llf > best.llf):
                best = res
        except Exception:
            continue
    if best is None:
        return {"ok": False}
    smp = np.asarray(best.smoothed_marginal_probabilities)
    fmp = np.asarray(best.filtered_marginal_probabilities)
    assign = smp.argmax(axis=1)
    means = [yv[assign == k].mean() if (assign == k).any() else -np.inf for k in range(2)]
    s = int(np.argmax(means))
    names = list(best.model.param_names)
    par = dict(zip(names, np.asarray(best.params)))
    P = np.asarray(best.regime_transition)[:, :, 0]          # P[i, j] = P(s_t = i | s_{t-1} = j)
    return {"ok": True, "smoothed": pd.Series(smp[:, s], index=idx),
            "filtered": pd.Series(fmp[:, s], index=idx),
            "mu_stress": float(par[f"const[{s}]"]), "mu_calm": float(par[f"const[{1 - s}]"]),
            "p_stay_stress": float(P[s, s]), "p_stay_calm": float(P[1 - s, 1 - s]),
            "sigma2": float(par.get("sigma2", np.nan)), "llf": float(best.llf), "stress_state": s}


def flag_threshold(prob: pd.Series, thr: float = 0.5) -> pd.Series:
    return (prob > thr).astype(float)


def flag_top_share(prob: pd.Series, share: float = BASE_SHARE) -> pd.Series:
    """Stress = the round(share * T) months with the highest probability (cr4_experiments.top_share)."""
    k = int(round(share * len(prob)))
    out = pd.Series(0.0, index=prob.index)
    out[prob.sort_values(ascending=False).index[:k]] = 1.0
    return out


# =============================================================================== OLS of Eq. (3)
def design(p: pd.DataFrame, stress, *, persistence: bool = True, extra: dict | None = None,
           outcome: str = OUTCOME, gprc: str = "gpr_country_z") -> dict:
    """Design of Eq. (3), identical to cr3_extra.interaction_fit: constant, GPRC, S, GPRC x S,
    log RV_t (if persistence), controls, extra columns, country dummies (first dropped)."""
    d = p.copy()
    d["S"] = d[stress] if isinstance(stress, str) else d.index.map(pd.Series(stress).to_dict())
    d["GxS"] = d[gprc] * d["S"]
    cols = [gprc, "S", "GxS"] + (["log_rv_t_z"] if persistence else []) + CTRL
    for name, fn in (extra or {}).items():
        d[name] = fn(d)
        cols.append(name)
    d = d.dropna(subset=[outcome] + cols)
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    Xdf = pd.concat([d[cols].reset_index(drop=True), cty.reset_index(drop=True)], axis=1).astype(float)
    X = np.column_stack([np.ones(len(d)), Xdf.values])
    return {"X": X, "y": d[outcome].values.astype(float), "names": ["const"] + list(Xdf.columns),
            "pos": 1 + cols.index("GxS"), "months": d.index.values, "country": d["country"].values,
            "S": d["S"].values.astype(float), "frame": d}


def ols_fit(X: np.ndarray, y: np.ndarray):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    return beta, y - X @ beta, XtX_inv


def se_hc3(X, resid, XtX_inv) -> np.ndarray:
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (resid / (1 - h))[:, None] ** 2).T @ X
    return np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))


def se_cluster(X, resid, XtX_inv, groups) -> np.ndarray:
    """statsmodels cov_type='cluster' with its default small-sample correction
    G/(G-1) * (N-1)/(N-K)."""
    codes, uniq = pd.factorize(np.asarray(groups))
    G, (n, k) = len(uniq), X.shape
    S = np.zeros((G, k))
    np.add.at(S, codes, X * resid[:, None])
    corr = G / (G - 1) * (n - 1) / (n - k)
    return np.sqrt(np.diag(XtX_inv @ (S.T @ S) @ XtX_inv) * corr)


def month_sums(v: np.ndarray, months) -> np.ndarray:
    """Sum the rows of v within each calendar month, months sorted ascending."""
    return pd.DataFrame(v, index=pd.Index(months)).groupby(level=0).sum().sort_index().values


def bartlett_lrv(S: np.ndarray, L: int) -> np.ndarray:
    """Bartlett long-run variance of the rows of S (T x k): S'S + sum_l w_l (G_l + G_l')."""
    Om = S.T @ S
    for lag in range(1, L + 1):
        G = S[lag:].T @ S[:-lag]
        Om = Om + (1 - lag / (L + 1)) * (G + G.T)
    return Om


def se_dk(X, resid, XtX_inv, months, L: int = 4) -> np.ndarray:
    """Driscoll-Kraay (identical to cr3_extra.driscoll_kraay / ms7): Bartlett HAC with L lags
    applied to the moment contributions summed by month; no small-sample correction."""
    S = month_sums(X * resid[:, None], months)
    return np.sqrt(np.diag(XtX_inv @ bartlett_lrv(S, L) @ XtX_inv))


# ------------------------------------------------------------------------------- fixed-b
def fixedb_null(T: int, L: int, n_sim: int = 100_000, seed: int = 20261013) -> np.ndarray:
    """Null distribution of |t| for a Bartlett-HAC (L lags) studentised mean of T iid N(0,1)
    observations. By fixed-b asymptotics (Kiefer & Vogelsang 2005) this is the reference
    distribution of a DK t-statistic with bandwidth b = (L+1)/T. Cached on disk."""
    path = RES / "cache" / f"fixedb_T{T}_L{L}.npy"
    if path.exists():
        return np.load(path)
    rng = np.random.RandomState(seed)
    out = np.empty(n_sim)
    chunk = 10_000
    w = 1 - np.arange(1, L + 1) / (L + 1)
    for a in range(0, n_sim, chunk):
        m = min(chunk, n_sim - a)
        e = rng.randn(m, T)
        u = e - e.mean(1, keepdims=True)
        om = (u ** 2).sum(1)
        for lag, wl in zip(range(1, L + 1), w):
            om += 2 * wl * (u[:, lag:] * u[:, :-lag]).sum(1)
        out[a:a + m] = np.abs(e.sum(1)) / np.sqrt(om)          # = sqrt(T)*mean / sqrt(om/T)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.sort(out))
    return np.sort(out)


def p_fixedb(t: float, T: int, L: int) -> float:
    null = fixedb_null(T, L)
    return float(1 - np.searchsorted(null, abs(t), side="left") / len(null))


# ------------------------------------------------------------------------------- bootstraps
def ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients via the normal equations (k x k solve). Used inside bootstrap loops:
    repeated np.linalg.lstsq calls on matrices of varying size leaked about 13 MB per 399
    calls on this machine (BLAS workspace), which exhausted RAM in the Monte Carlo workers.
    Falls back to lstsq if X'X is singular."""
    try:
        return np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def _cluster_index(groups):
    codes, uniq = pd.factorize(np.asarray(groups), sort=True)
    return codes, len(uniq)


TIE_RTOL = 1e-9


def wild_cluster_t(X, y, groups, j, weights: str = "rademacher", B: int = 399, seed: int = 42,
                   enumerate_all: bool = False):
    """t_obs and bootstrap t* of the restricted wild cluster bootstrap-t (see wild_cluster_p)."""
    codes, G = _cluster_index(groups)
    n, k = X.shape
    beta, resid, XtX_inv = ols_fit(X, y)
    corr = G / (G - 1) * (n - 1) / (n - k)
    w = (X @ XtX_inv)[:, j]
    S_obs = np.zeros(G)
    np.add.at(S_obs, codes, w * resid)
    t_obs = beta[j] / np.sqrt(corr * np.sum(S_obs ** 2))
    keep = [c for c in range(k) if c != j]
    Xr = X[:, keep]
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    yhat_r, u_r = Xr @ br, y - Xr @ br
    if enumerate_all:
        V = np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
    else:
        rng = np.random.RandomState(seed)
        V = rng.choice(WEBB6 if weights == "webb" else np.array([-1.0, 1.0]), size=(B, G))
    Ystar = yhat_r[:, None] + u_r[:, None] * V[:, codes].T          # n x B
    Bstar = XtX_inv @ (X.T @ Ystar)                                    # k x B
    Ustar = Ystar - X @ Bstar
    Sg = np.zeros((G, Ystar.shape[1]))
    np.add.at(Sg, codes, w[:, None] * Ustar)
    t_star = Bstar[j] / np.sqrt(corr * np.sum(Sg ** 2, axis=0))
    return float(t_obs), t_star


def wild_cluster_p(X, y, groups, j, weights: str = "rademacher", B: int = 399, seed: int = 42,
                   enumerate_all: bool = False) -> float:
    """Restricted wild cluster bootstrap-t (Cameron, Gelbach & Miller 2008) for H0: beta_j = 0,
    with statsmodels-corrected cluster SEs, vectorised over draws. Weights are drawn with
    RandomState(seed).choice(..., size=(B, G)), i.e. the same stream as ms7's draw-by-draw loop.
    enumerate_all=True uses all 2^G Rademacher sign patterns (exact for G = 5).

    Ties: the sign patterns in which all clusters get the same weight reproduce +/- t_obs
    exactly (up to rounding). They are counted as |t*| >= |t_obs| (relative tolerance
    TIE_RTOL). ms7 compared raw floats, so whether a tie was counted depended on rounding;
    with G = 5 Rademacher weights the smallest attainable p-value is 2/32 = 0.0625."""
    t_obs, t_star = wild_cluster_t(X, y, groups, j, weights, B, seed, enumerate_all)
    return float(np.mean(np.abs(t_star) >= abs(t_obs) * (1 - TIE_RTOL)))


def mbb_ols(X, y, months, j, B: int = 499, block: int = 6, seed: int = 42, s_col=None):
    """Moving-block bootstrap over calendar months (ms7.block_bootstrap): blocks of `block`
    consecutive months, T months per draw, OLS re-estimated; SE = sd of draws and
    p = 2 min(share >= 0, share <= 0). Draws whose S x GPRC column has < 3 distinct values
    are skipped (as in ms7)."""
    months = np.asarray(months)
    dates = np.sort(pd.unique(months))
    T = len(dates)
    # integer month positions: hashing numpy.datetime64 dict keys leaked memory (NumPy 2.x)
    mpos = np.searchsorted(dates, months)
    rows_of = [np.where(mpos == t)[0] for t in range(T)]
    nblocks = int(np.ceil(T / block))
    rng = np.random.RandomState(seed)
    draws = []
    gxs = X[:, j] if s_col is None else s_col
    for _ in range(B):
        starts = rng.randint(0, T - block + 1, size=nblocks)
        sel = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        rows = np.concatenate([rows_of[t] for t in sel])
        if len(np.unique(gxs[rows])) < 3:
            continue
        draws.append(ols_coef(X[rows], y[rows])[j])
    draws = np.array(draws)
    return float(draws.std(ddof=1)), float(2 * min((draws >= 0).mean(), (draws <= 0).mean())), draws


# ------------------------------------------------------------------------------- OLS bundle
def ols_inference(dz: dict, *, full: bool = True, dk_lags=DK_LAGS, B_webb: int = 399,
                  B_month: int = 399, B_mbb: int = 399, seed: int = 42) -> dict:
    """beta_3 of Eq. (3) with its standard errors / p-values.
    full=False: DK(4) only (cheap; used for secondary regime flags in the Monte Carlo)."""
    X, y, j, months = dz["X"], dz["y"], dz["pos"], dz["months"]
    beta, resid, XtX_inv = ols_fit(X, y)
    b = float(beta[j])
    T = len(np.unique(months))
    out = {"b3": b, "n": int(len(y)), "T": T}
    for L in (dk_lags if full else (4,)):
        se = se_dk(X, resid, XtX_inv, months, L)[j]
        out[f"dk{L}"] = {"se": float(se), "p": pval(b, se)}
    if not full:
        return out
    out["dk4_fixedb"] = {"p": p_fixedb(b / out["dk4"]["se"], T, 4)}
    se = se_hc3(X, resid, XtX_inv)[j]
    out["hc3"] = {"se": float(se), "p": pval(b, se)}
    se = se_cluster(X, resid, XtX_inv, months)[j]
    out["cl_month"] = {"se": float(se), "p": pval(b, se)}
    out["wcb_rademacher_enum"] = {"p": wild_cluster_p(X, y, dz["country"], j, enumerate_all=True)}
    out["wcb_webb"] = {"p": wild_cluster_p(X, y, dz["country"], j, "webb", B_webb, seed)}
    out["wb_month"] = {"p": wild_cluster_p(X, y, months, j, "rademacher", B_month, seed)}
    se, p, _ = mbb_ols(X, y, months, j, B=B_mbb, seed=seed)
    out["mbb"] = {"se": se, "p": p}
    return out


# =============================================================================== DML
class AvgEnsemble:
    """Average of the predictions of several learners from src.models.dml._make_model
    (prespec P6: Lasso + random forest + XGBoost). Minimal sklearn-compatible API."""

    def __init__(self, names=("lasso", "rf", "xgb")):
        self.names = tuple(names)

    def get_params(self, deep=True):
        return {"names": self.names}

    def set_params(self, **kw):
        self.names = tuple(kw.get("names", self.names))
        return self

    def fit(self, X, y):
        self.models_ = [_make_model(n).fit(X, y) for n in self.names]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def learner(name: str):
    """'lasso' | 'ridge' | 'rf' | 'xgb' (the paper's factory) or 'ensemble'."""
    return AvgEnsemble() if name == "ensemble" else _make_model(name)


def lasso_fixed(alpha: float) -> Pipeline:
    """Lasso with a fixed penalty (used in the Monte Carlo for speed; prespec P4)."""
    return Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=5000))])


def dml_frame(p: pd.DataFrame, stress_col: str):
    """cr4_experiments.dml_frame: treatment GxS; controls GPRC, S, log RV_t, CTRL, country dummies."""
    d = p.copy()
    d["GxS"] = d["gpr_country_z"] * d[stress_col]
    cty = pd.get_dummies(d["country"], prefix="cty", drop_first=True).astype(float)
    d = pd.concat([d, cty], axis=1)
    return d, ["gpr_country_z", stress_col, "log_rv_t_z"] + CTRL + list(cty.columns)


def make_splits(dates, scheme: str, K: int = 5, seed: int = 0, gap: int = 0, nbr: int = 0):
    """Cross-fitting folds.
    shuffled  : KFold(K, shuffle=True, random_state=seed) over rows (as src _cross_fit)
    blocked   : K contiguous calendar blocks; train on strictly earlier months; first block
                not validated (src make_blocked_splits)
    blocked_gap: as blocked, but the `gap` months just before the validation block are
                removed from training (embargo)
    neighbour : validation = block; train = all months outside [block - nbr, block + nbr]
                (neighbour deletion, Semenova et al. 2023)
    reverse   : reverse cross-fitting (Ciganovic, D'Amario & Tancioni 2026, sect. 2.2): K
                adjacent blocks, every block validated; nuisances trained on the larger side of
                the remaining sample - later blocks for the first half, earlier blocks for the
                second half, both sides for the central block when K is odd; no buffer."""
    dates = np.asarray(dates)
    if scheme == "shuffled":
        return list(KFold(n_splits=K, shuffle=True, random_state=seed).split(np.zeros(len(dates))))
    if scheme == "blocked":
        return make_blocked_splits(dates, K)
    uniq = np.array(sorted(pd.unique(dates)))
    pos = {d: i for i, d in enumerate(uniq)}
    mpos = np.array([pos[d] for d in dates])
    blocks = np.array_split(np.arange(len(uniq)), K)
    out = []
    for k, blk in enumerate(blocks, start=1):
        lo, hi = blk.min(), blk.max()
        val = np.where((mpos >= lo) & (mpos <= hi))[0]
        if scheme == "blocked_gap":
            train = np.where(mpos < lo - gap)[0]
        elif scheme == "neighbour":
            train = np.where((mpos < lo - nbr) | (mpos > hi + nbr))[0]
        elif scheme == "reverse":
            later, earlier = mpos > hi, mpos < lo
            if K % 2 == 0:
                train = np.where(later if k <= K // 2 else earlier)[0]
            else:
                mid = (K + 1) // 2
                train = np.where(later if k < mid else earlier if k > mid else (later | earlier))[0]
        else:
            raise ValueError(scheme)
        if len(train) and len(val):
            out.append((train, val))
    return out


def cross_fit(Y, X, model, splits) -> np.ndarray:
    """Residuals Y - E_hat[Y|X] on validation folds (NaN for rows never validated)."""
    res = np.full(len(Y), np.nan)
    for tr, va in splits:
        m = clone(model)
        m.fit(X[tr], Y[tr])
        res[va] = Y[va] - m.predict(X[va])
    return res


def dml_theta(Y, D, X, dates, scheme: str = "blocked", K: int = 5, model_y=None, model_d=None,
              reps: int | None = None, seed: int = 42, dk_lags=(4, 6, 12), pool: str | None = None,
              **split_kw) -> dict:
    """Partially linear DML (cr4_experiments.dml_dk). theta = median over repetitions of
    sum(eD eY)/sum(eD^2); shuffled uses 3 repetitions with seeds seed, seed+100, seed+200.
    SEs: i.i.d. score variance, and DK-type: scores psi = eD (eY - theta eD) summed by
    month, Bartlett HAC with L lags, divided by sum(eD^2)."""
    model_y = model_y if model_y is not None else _make_model("lasso")
    model_d = model_d if model_d is not None else _make_model("lasso")
    reps = reps if reps is not None else (3 if scheme == "shuffled" else 1)
    n = len(Y)
    pool = pool if pool is not None else ("fold_mean" if scheme == "reverse" else "pooled")
    eY_all, eD_all, th, fold_th = np.full((n, reps), np.nan), np.full((n, reps), np.nan), [], []
    for r in range(reps):
        s = seed + r * 100
        splits = make_splits(dates, scheme, K, seed=s, **split_kw)
        eY, eD = cross_fit(Y, X, model_y, splits), cross_fit(D, X, model_d, splits)
        v = ~(np.isnan(eY) | np.isnan(eD))
        if pool == "fold_mean":
            # Ciganovic et al. (2026, eq. 2.5): average of the fold-specific slopes
            fk = [float(np.dot(eD[va], eY[va]) / np.dot(eD[va], eD[va])) for _, va in splits]
            fold_th.append(fk)
            th.append(np.mean(fk))
        else:
            th.append(np.dot(eD[v], eY[v]) / np.dot(eD[v], eD[v]))
        eY_all[:, r], eD_all[:, r] = eY, eD
    theta = float(np.median(th))
    eY, eD = np.nanmean(eY_all, axis=1), np.nanmean(eD_all, axis=1)
    v = ~(np.isnan(eY) | np.isnan(eD))
    eY, eD, dv = eY[v], eD[v], np.asarray(dates)[v]
    psi = eD * (eY - theta * eD)
    J = np.sum(eD ** 2)
    se_iid = np.sqrt(np.mean(psi ** 2) / (len(psi) * np.mean(eD ** 2) ** 2))
    out = {"theta": theta, "n": int(v.sum()), "scheme": scheme, "K": K, "pool": pool,
           "fold_thetas": fold_th[0] if fold_th else None,
           "iid": {"se": float(se_iid), "p": pval(theta, se_iid)},
           # out-of-fold R^2 of the two nuisance regressions (diagnostic)
           "r2_y": float(1 - np.mean(eY ** 2) / np.var(np.asarray(Y)[v])),
           "r2_d": float(1 - np.mean(eD ** 2) / np.var(np.asarray(D)[v]))}
    S = month_sums(psi[:, None], dv)
    for L in dk_lags:
        se = float(np.sqrt(bartlett_lrv(S, L)[0, 0]) / J)
        out[f"dk{L}"] = {"se": se, "p": pval(theta, se)}
    return out


def dml_by_regime(Y, G, S, X, dates, scheme: str = "blocked", K: int = 5, model_y=None, model_g=None,
                  reps: int | None = None, seed: int = 42, L: int = 4, wild_B: int = 0) -> dict:
    """Interacted partially linear DML: Y = theta_c G (1-S) + theta_s G S + g(X) + e, with the
    regime S inside X and G NOT a function of X. Y and G are cross-fitted on X; theta_c and
    theta_s are the residual slopes in calm and stress observations; the state dependence is
    delta = theta_s - theta_c (the DML analogue of beta_3).
    Why: with GPRC x Stress as the treatment and GPRC and Stress among the controls, the
    treatment is an exact function of the controls, so a flexible learner can predict it
    (R2 of 0.7-0.9 for RF/XGB) and the treatment-as-interaction design is not identified.
    Variance: per-observation influence psi_s/J_s - psi_c/J_c, summed by month, Bartlett HAC."""
    model_y = model_y if model_y is not None else _make_model("lasso")
    model_g = model_g if model_g is not None else _make_model("lasso")
    reps = reps if reps is not None else (3 if scheme == "shuffled" else 1)
    S = np.asarray(S, dtype=float)
    out_reps, eY_all, eG_all = [], [], []
    for r in range(reps):
        splits = make_splits(dates, scheme, K, seed=seed + r * 100)
        eY, eG = cross_fit(Y, X, model_y, splits), cross_fit(G, X, model_g, splits)
        v = ~(np.isnan(eY) | np.isnan(eG))
        th_c = np.dot(eG[v] * (1 - S[v]), eY[v]) / np.dot(eG[v] * (1 - S[v]), eG[v])
        th_s = np.dot(eG[v] * S[v], eY[v]) / np.dot(eG[v] * S[v], eG[v])
        out_reps.append((th_s - th_c, th_c, th_s))
        eY_all.append(eY)
        eG_all.append(eG)
    k = int(np.argsort([x[0] for x in out_reps])[len(out_reps) // 2])     # median repetition
    delta, th_c, th_s = out_reps[k]
    eY, eG = np.nanmean(eY_all, axis=0), np.nanmean(eG_all, axis=0)
    v = ~(np.isnan(eY) | np.isnan(eG))
    eY, eG, Sv, dv = eY[v], eG[v], S[v], np.asarray(dates)[v]
    Jc, Js = np.sum(eG ** 2 * (1 - Sv)), np.sum(eG ** 2 * Sv)
    infl = eG * Sv * (eY - th_s * eG) / Js - eG * (1 - Sv) * (eY - th_c * eG) / Jc
    se = float(np.sqrt(bartlett_lrv(month_sums(infl[:, None], dv), L)[0, 0]))
    out = {"delta": float(delta), "theta_calm": float(th_c), "theta_stress": float(th_s),
           f"dk{L}": {"se": se, "p": pval(delta, se)}, "n": int(v.sum()), "scheme": scheme,
           "r2_y": float(1 - np.mean(eY ** 2) / np.var(np.asarray(Y)[v])),
           "r2_g": float(1 - np.mean(eG ** 2) / np.var(np.asarray(G)[v]))}
    if wild_B:
        # wild bootstrap clustered by month on the final stage (nuisance residuals held fixed):
        # eY = theta_c eG + delta (eG S) + u, restricted fit imposes delta = 0
        Z = np.column_stack([eG, eG * Sv])
        out["wild_month"] = {"p": wild_cluster_p(Z, eY, dv, 1, "rademacher", wild_B, seed), "B": wild_B}
    return out


def dml_interaction(p: pd.DataFrame, stress_col: str, scheme: str = "blocked", **kw) -> dict:
    d, ctrl = dml_frame(p, stress_col)
    d = d.dropna(subset=[OUTCOME, "GxS"] + ctrl)
    return dml_theta(d[OUTCOME].values.astype(float), d["GxS"].values.astype(float),
                     d[ctrl].values.astype(float), d.index.values, scheme=scheme, **kw)


# =============================================================================== regime constructions
def loco_flags(p: pd.DataFrame, starts: int = 10, rule: str = "top") -> pd.Series:
    """Leave-one-country-out regime (cr4_experiments.block_loco): market i is classified by a
    2-state MS on the average log RV of the other markets; rule 'top' keeps the baseline
    stress share, rule 'p05' uses P > 0.5. Returns a row-aligned 0/1 series."""
    out = pd.Series(np.nan, index=range(len(p)))
    cc = p["country"].values
    for c in CC5:
        reg = regional_mean(p, exclude=c)
        fm = fit_ms(reg, starts=starts)
        fl = flag_top_share(fm["smoothed"]) if rule == "top" else flag_threshold(fm["smoothed"])
        mask = cc == c
        out[np.where(mask)[0]] = p.index[mask].map(fl.to_dict()).values
    return pd.Series(out.values, index=p.index)


MIN_STRESS_MONTHS = 5


def ms_regime_bootstrap(p: pd.DataFrame, B: int = 199, block: int = 6, seed: int = 42,
                        starts: int = 3, rule: str = "p05") -> dict:
    """Moving-block bootstrap over months that RE-ESTIMATES the Markov-switching regime in
    every draw, so the SE of beta_3 includes regime-estimation (generated-regressor) error.
    Each draw concatenates blocks of `block` months; the regional average log RV of the
    resampled sequence is re-fitted by MS; stress flags follow `rule`; Eq. (3) is re-estimated
    on the resampled rows (z-scores kept fixed). Returns SE, percentile p and the draws."""
    dz = design(p, STRESS)
    d = dz["frame"]
    months = np.sort(np.asarray(d.index.unique()))
    T = len(months)
    mpos = np.searchsorted(months, d.index.values)                 # integer positions (see mbb_ols)
    rows_list = [np.where(mpos == t)[0] for t in range(T)]
    reg = regional_mean(p).reindex(months)
    Xbase = dz["X"].copy()
    jS, jG, j = dz["names"].index("S"), dz["names"].index("gpr_country_z"), dz["pos"]
    rng = np.random.RandomState(seed)
    nblocks = int(np.ceil(T / block))
    draws, degenerate = [], 0
    for _ in range(B):
        starts_ = rng.randint(0, T - block + 1, size=nblocks)
        seq = np.concatenate([np.arange(s, s + block) for s in starts_])[:T]
        fm = fit_ms(pd.Series(reg.values[seq]), starts=starts)
        if not fm["ok"]:
            continue
        prob = fm["smoothed"].values
        flag = (prob > 0.5).astype(float) if rule == "p05" else \
            flag_top_share(pd.Series(prob)).values
        # degenerate MS solutions (almost no stress months, or a 'stress' state covering most of
        # the sample) make Eq. (3) nearly singular and produce exploding draws: skip and count
        if flag.sum() < MIN_STRESS_MONTHS or flag.sum() > T / 2:
            degenerate += 1
            continue
        rows = np.concatenate([rows_list[t] for t in seq])
        sflag = np.concatenate([np.full(len(rows_list[t]), flag[k]) for k, t in enumerate(seq)])
        Xb = Xbase[rows].copy()
        Xb[:, jS] = sflag
        Xb[:, j] = Xb[:, jG] * sflag
        if len(np.unique(Xb[:, j])) < 3:
            continue
        draws.append(ols_coef(Xb, dz["y"][rows])[j])
    draws = np.array(draws)
    q25, q75 = np.percentile(draws, [25, 75])
    return {"se": float(draws.std(ddof=1)), "se_iqr": float((q75 - q25) / 1.349),
            "p": float(2 * min((draws >= 0).mean(), (draws <= 0).mean())),
            "B_used": int(len(draws)), "degenerate_skipped": int(degenerate), "draws": draws}
