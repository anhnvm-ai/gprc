"""Forecast comparison tests: Diebold-Mariano, Giacomini-White, Model Confidence Set."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import norm

def _loss(e, kind="se"):
    e = np.asarray(e)
    return e ** 2 if kind == "se" else np.abs(e)

def diebold_mariano(y, yh1, yh2, h: int = 1, loss: str = "se") -> dict:
    e1, e2 = np.asarray(y) - np.asarray(yh1), np.asarray(y) - np.asarray(yh2)
    d = _loss(e1, loss) - _loss(e2, loss)
    n = len(d); mean_d = d.mean()
    # Newey-West variance up to lag h-1
    var = np.var(d, ddof=1)
    for k in range(1, h):
        cov = np.cov(d[k:], d[:-k])[0, 1]
        var += 2 * (1 - k / h) * cov
    se = np.sqrt(var / n)
    dm = mean_d / se
    pval = 2 * (1 - norm.cdf(abs(dm)))
    return {"DM_stat": float(dm), "p_value": float(pval), "mean_loss_diff": float(mean_d)}

def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.10, B: int = 5000, seed: int = 0) -> dict:
    """Hansen-Lunde-Nason MCS using stationary bootstrap (block ~12).
    losses: DataFrame of per-period loss for each model (columns = model names).
    Returns set of models that are not significantly worse than the best at level alpha.
    """
    from arch.bootstrap import MCS
    mcs = MCS(losses.values, size=alpha, reps=B, block_size=12, method="R", seed=seed)
    mcs.compute()
    cols = list(losses.columns)

    # `included` and `excluded` may be Series of integer labels or boolean masks
    inc = mcs.included
    if isinstance(inc, pd.Series): inc = inc.values
    inc = np.asarray(inc).ravel()
    if inc.dtype == bool:
        included = [cols[i] for i in np.where(inc)[0]]
    else:
        included = [cols[int(i)] for i in inc]

    # `pvalues` may be DataFrame indexed by model id with column 'Pvalue'
    pv = mcs.pvalues
    pv_map = {}
    if isinstance(pv, pd.DataFrame):
        col = pv.columns[0]
        for label, val in pv[col].items():
            try: pv_map[cols[int(label)]] = float(val)
            except (ValueError, IndexError): pv_map[str(label)] = float(val)
    elif isinstance(pv, pd.Series):
        for label, val in pv.items():
            try: pv_map[cols[int(label)]] = float(val)
            except (ValueError, IndexError): pv_map[str(label)] = float(val)
    else:
        for i, val in enumerate(np.asarray(pv).ravel()):
            pv_map[cols[i]] = float(val)
    return {"included_models": included, "pvalues": pv_map}
