"""Shared helper — build supervised tensors and split temporally.

Used by phases 3, 4, 5, 6 so we don't duplicate feature engineering.

Public API:
    prepare(target, train_frac, val_frac, target_kind)  -> Prepared (single split, scaled)
    prepare_raw(target, target_kind)                    -> RawPanel (unscaled, no split — for walk-forward)
    fit_scale(X_tr, X_va, X_te)                         -> (X_tr_s, X_va_s, X_te_s, scaler)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd

from src.utils.io import load_config, ROOT
from src.features.features import add_lags, add_rolling, add_interactions, make_supervised


@dataclass
class Prepared:
    X_tr: np.ndarray; y_tr: np.ndarray
    X_va: np.ndarray; y_va: np.ndarray
    X_te: np.ndarray; y_te: np.ndarray
    idx_tr: pd.DatetimeIndex
    idx_va: pd.DatetimeIndex
    idx_te: pd.DatetimeIndex
    inter_tr: np.ndarray | None
    inter_va: np.ndarray | None
    inter_te: np.ndarray | None
    feature_cols: list[str]
    target: str


@dataclass
class RawPanel:
    X: np.ndarray              # (N, L, F) unscaled
    y: np.ndarray              # (N,)
    idx: pd.DatetimeIndex      # (N,)
    inter: np.ndarray | None   # (N,)
    feature_cols: list[str]
    target: str
    target_kind: str


def _build_target(df: pd.DataFrame, base_target: str, kind: str) -> tuple[pd.Series, str]:
    """kind='return' uses base_target as-is; kind='vol' returns next-day-ahead 21d realized vol."""
    if kind == "return":
        return df[base_target], base_target
    if kind == "vol":
        # next-period realized volatility (annualized) — derived from base return column
        ret_col = base_target if base_target.endswith("_ret") else base_target + "_ret"
        if ret_col not in df.columns:
            raise KeyError(f"return column {ret_col!r} required to build volatility target")
        rv = df[ret_col].rolling(21).std() * np.sqrt(252)
        # 1-day-ahead forecast target = next bar's 21d realized vol
        target = rv.shift(-1)
        new_name = ret_col.replace("_ret", "_rv21_fwd1")
        target.name = new_name
        return target, new_name
    raise ValueError(f"Unknown target kind: {kind!r}")


def prepare_raw(target: str = "VNINDEX_ret", target_kind: str = "return") -> RawPanel:
    """Build supervised arrays (unscaled). Used by walk-forward and seed-loop runners."""
    cfg = load_config()
    df = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "final_dataset.parquet").dropna(how="all")
    feature_cols = [c for c in df.columns if c != target and df[c].dtype.kind in "fiu"]
    df = df[[target] + feature_cols].copy()

    # Build the actual target series + name
    target_series, tname = _build_target(df, target, target_kind)
    df[tname] = target_series

    feat = add_lags(df, feature_cols, cfg["features"]["lags"])
    feat = add_rolling(feat, feature_cols, windows=(5, 21))
    feat = add_interactions(feat, [tuple(p) for p in cfg["features"]["interaction_terms"]])
    feat = feat.dropna()

    use_cols = [c for c in feat.columns if c != tname]
    X3d, y, idx = make_supervised(feat[use_cols + [tname]], tname, cfg["features"]["lookback"])

    X3d = np.nan_to_num(X3d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y   = np.nan_to_num(y,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    inter_col = "GPR_x_dFX" if "GPR_x_dFX" in feat.columns else None
    if inter_col:
        inter = feat[inter_col].reindex(idx).values.astype(np.float32)
        inter = np.nan_to_num(inter, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        inter = None

    return RawPanel(X=X3d, y=y, idx=idx, inter=inter,
                    feature_cols=use_cols, target=tname, target_kind=target_kind)


def fit_scale(X_tr: np.ndarray, *others: np.ndarray, clip: float = 8.0):
    from sklearn.preprocessing import StandardScaler
    n_feat = X_tr.shape[2]
    scaler = StandardScaler().fit(X_tr.reshape(-1, n_feat))
    def _t(a):
        s = scaler.transform(a.reshape(-1, n_feat)).reshape(a.shape)
        return np.clip(s, -clip, clip).astype(np.float32)
    return (_t(X_tr), *[_t(a) for a in others], scaler)


def prepare(target: str = "VNINDEX_ret",
            train_frac: float = 0.70,
            val_frac:   float = 0.15,
            target_kind: str = "return") -> Prepared:
    """Single train/val/test split with train-only scaling (used by phases 3-6)."""
    raw = prepare_raw(target=target, target_kind=target_kind)
    X3d, y, idx, inter = raw.X, raw.y, raw.idx, raw.inter

    n = len(idx)
    cut1 = int(n * train_frac)
    cut2 = int(n * (train_frac + val_frac))

    X_tr_s, X_va_s, X_te_s, _ = fit_scale(X3d[:cut1], X3d[cut1:cut2], X3d[cut2:])

    if inter is not None:
        mu, sd = inter[:cut1].mean(), inter[:cut1].std() + 1e-8
        inter_s = np.clip((inter - mu) / sd, -8, 8).astype(np.float32)
        inter_tr, inter_va, inter_te = inter_s[:cut1], inter_s[cut1:cut2], inter_s[cut2:]
    else:
        inter_tr = inter_va = inter_te = None

    return Prepared(
        X_tr=X_tr_s, y_tr=y[:cut1],
        X_va=X_va_s, y_va=y[cut1:cut2],
        X_te=X_te_s, y_te=y[cut2:],
        idx_tr=idx[:cut1], idx_va=idx[cut1:cut2], idx_te=idx[cut2:],
        inter_tr=inter_tr, inter_va=inter_va, inter_te=inter_te,
        feature_cols=raw.feature_cols, target=raw.target,
    )
