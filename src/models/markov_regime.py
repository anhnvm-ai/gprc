"""Markov-Switching regime identification helpers.

Robust 2-state Markov-Switching estimation on log realized volatility.
Used by the regime-dependent channel analysis (ms1..ms6 scripts).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")


@dataclass
class RegimeResult:
    smoothed_stress: pd.Series      # P(stress regime) per date
    stress_flag: pd.Series          # 1 if P(stress) > 0.5
    mu_calm: float
    mu_stress: float
    p_stay_calm: float
    p_stay_stress: float
    dur_calm: float                 # expected duration (months)
    dur_stress: float
    switching_variance: bool
    llf: float
    n_stress: int
    n_total: int
    converged: bool = True


def fit_markov_2state(y: pd.Series, *, switching_variance_try=False,
                      n_restarts: int = 10, seed: int = 42) -> RegimeResult:
    """Fit the main 2-state Markov-Switching model on series y (e.g. log_rv).

    Main specification (default): two-state SWITCHING-MEAN model with CONSTANT
    variance (``switching_variance_try=False``). This fixes the main pipeline to
    the specification reported in the paper and removes any dependence on
    convergence-driven fallback. Random restarts guard against local optima.

    Set ``switching_variance_try=True`` only for the Appendix robustness check,
    which prefers a switching-variance fit and falls back to switching-mean if
    its EM fails to converge. The 'stress' regime is the higher-mean state.
    """
    y = y.dropna()
    yv = np.asarray(y, dtype=float)
    yv_idx = y.index

    sv_order = [True, False] if switching_variance_try else [False]
    best = None
    for sv in sv_order:
        for rep in range(n_restarts):
            try:
                mod = MarkovRegression(yv, k_regimes=2, trend="c",
                                       switching_variance=sv)
                if rep == 0:
                    res = mod.fit(em_iter=40, maxiter=200)
                else:
                    rng = np.random.RandomState(seed + rep)
                    sp0 = mod.start_params * (1 + 0.15 * rng.randn(len(mod.start_params)))
                    res = mod.fit(start_params=sp0, em_iter=40, maxiter=200)
                if np.isfinite(res.llf) and (best is None or res.llf > best[1]):
                    best = (sv, res.llf, res)
            except Exception:
                continue
        if best is not None:
            break

    if best is None:
        raise RuntimeError("Markov-switching estimation failed for all configs")

    sv, llf, res = best
    smp = np.asarray(res.smoothed_marginal_probabilities)   # (T, 2)
    assign = smp.argmax(axis=1)
    means_by_reg = [yv[assign == k].mean() if (assign == k).any() else -np.inf
                    for k in range(2)]
    stress_idx = int(np.argmax(means_by_reg))
    calm_idx   = 1 - stress_idx

    sp = pd.Series(smp[:, stress_idx], index=yv_idx, name="p_stress")
    flag = (sp > 0.5).astype(int).rename("stress")

    # regime_transition shape: (k, k, 1) ; [i,j] = P(s_t=i | s_{t-1}=j) or transpose?
    # statsmodels: regime_transition[i, j] = P(s_t=j | s_{t-1}=i) -> diag is stay prob
    rt = np.asarray(res.regime_transition)[:, :, 0]
    p_stay_stress = float(rt[stress_idx, stress_idx])
    p_stay_calm   = float(rt[calm_idx, calm_idx])

    def _dur(p):  # expected duration in months
        return 1.0 / (1.0 - p) if p < 1 else np.inf

    return RegimeResult(
        smoothed_stress=sp,
        stress_flag=flag,
        mu_calm=float(means_by_reg[calm_idx]),
        mu_stress=float(means_by_reg[stress_idx]),
        p_stay_calm=p_stay_calm,
        p_stay_stress=p_stay_stress,
        dur_calm=_dur(p_stay_calm),
        dur_stress=_dur(p_stay_stress),
        switching_variance=sv,
        llf=float(llf),
        n_stress=int(flag.sum()),
        n_total=int(len(flag)),
    )
