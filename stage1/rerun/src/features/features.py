"""Feature engineering: lags, rolling stats, interactions, scaling helpers."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in df.columns: continue
        for L in lags:
            out[f"{c}_lag{L}"] = df[c].shift(L)
    return out

def add_rolling(df: pd.DataFrame, cols: list[str], windows=(5, 21)) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in df.columns: continue
        for w in windows:
            out[f"{c}_ma{w}"]  = df[c].rolling(w).mean()
            out[f"{c}_std{w}"] = df[c].rolling(w).std()
    return out

def add_interactions(df: pd.DataFrame, pairs: list[tuple[str,str]]) -> pd.DataFrame:
    out = df.copy()
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            out[f"{a}_x_{b}"] = df[a] * df[b]
    return out

def make_supervised(df: pd.DataFrame, target: str, lookback: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """3D tensor (N, L, F) for sequence models + 1D y aligned to last timestep."""
    feats = [c for c in df.columns if c != target]
    X, y, idx = [], [], []
    arr = df[feats].values; tar = df[target].values
    for t in range(lookback, len(df)):
        if np.isnan(arr[t-lookback:t]).any() or np.isnan(tar[t]): continue
        X.append(arr[t-lookback:t]); y.append(tar[t]); idx.append(df.index[t])
    return np.asarray(X), np.asarray(y), pd.DatetimeIndex(idx)

def get_scaler(name: str = "standard"):
    return {"standard": StandardScaler, "minmax": MinMaxScaler}[name]()
