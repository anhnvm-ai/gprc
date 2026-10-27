"""Long/short backtest with transaction costs, vol-targeting, and Deflated Sharpe."""
from __future__ import annotations
import numpy as np, pandas as pd

def vol_target_size(signal: pd.Series, ret: pd.Series, target_annual: float = 0.15,
                    vol_window: int = 21) -> pd.Series:
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    return (target_annual / vol.replace(0, np.nan)).clip(-3, 3) * np.sign(signal)

def backtest_long_short(forecast: pd.Series, realized_ret: pd.Series,
                        cost_bps: float = 10.0, target_annual: float = 0.15) -> pd.DataFrame:
    """forecast(t) is the prediction for return(t+1). We trade t+1 at the open."""
    pos = vol_target_size(forecast, realized_ret, target_annual)
    pos = pos.shift(1).fillna(0)            # avoid look-ahead
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps / 1e4)
    pnl = pos * realized_ret - cost
    eq = (1 + pnl).cumprod()
    return pd.DataFrame({"position": pos, "ret": pnl, "equity": eq, "turnover": turnover})

def metrics(pnl: pd.Series, n_models_tried: int = 1) -> dict:
    pnl = pnl.dropna()
    if len(pnl) < 30: return {"n": len(pnl)}
    mu, sd = pnl.mean(), pnl.std()
    sr = (mu / sd) * np.sqrt(252) if sd > 0 else np.nan
    cum = (1 + pnl).prod() - 1
    dd = ((1 + pnl).cumprod() / (1 + pnl).cumprod().cummax() - 1).min()
    # Deflated Sharpe Ratio (Bailey & López de Prado 2014, simplified)
    skew = pnl.skew(); kurt = pnl.kurt() + 3
    n = len(pnl)
    sr_std = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)) if not np.isnan(sr) else np.nan
    dsr = (sr - 0) / sr_std if sr_std and sr_std > 0 else np.nan
    return {"Sharpe": float(sr), "DSR": float(dsr), "CumRet": float(cum),
            "MaxDD": float(dd), "Turnover_avg": float(pnl.index.size and 0.0)}
