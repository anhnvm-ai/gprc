"""Targets: log-return, realized vol, GARCH(1,1) conditional vol."""
from __future__ import annotations
import numpy as np, pandas as pd

def log_return(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1))

def realized_vol(ret: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    rv = ret.rolling(window).std()
    return rv * np.sqrt(252) if annualize else rv

def garch_vol(ret: pd.Series, refit_idx: pd.DatetimeIndex | None = None) -> pd.Series:
    """One-step-ahead conditional volatility from GARCH(1,1)-t. Refit at refit_idx (default monthly)."""
    from arch import arch_model
    ret = ret.dropna() * 100  # arch wants %
    if refit_idx is None:
        refit_idx = pd.date_range(ret.index[252], ret.index[-1], freq="M")
    sigma = pd.Series(index=ret.index, dtype=float)
    last_anchor = ret.index[0]
    for anchor in refit_idx:
        train = ret.loc[:anchor]
        if len(train) < 252: continue
        am = arch_model(train, vol="Garch", p=1, q=1, dist="t")
        res = am.fit(disp="off")
        cond = res.conditional_volatility
        sigma.loc[cond.index] = cond.values
    sigma = sigma / 100.0
    return sigma * np.sqrt(252)
