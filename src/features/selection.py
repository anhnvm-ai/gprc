"""Feature selection: Lasso path (stability) + lightweight Boruta-style permutation test.

Returns ranked list of feature names with selection status.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def lasso_select(X: pd.DataFrame, y: np.ndarray, n_repeats: int = 20,
                 sample_frac: float = 0.8, seed: int = 0) -> pd.Series:
    """Stability selection: fraction of subsamples in which a feature receives nonzero Lasso coef."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(X.shape[1], dtype=int)
    n = len(X); m = int(n * sample_frac)
    sc = StandardScaler().fit(X.values)
    Xs = sc.transform(X.values)
    for _ in range(n_repeats):
        idx = rng.choice(n, size=m, replace=False)
        las = LassoCV(cv=3, n_alphas=20, max_iter=2000, n_jobs=-1).fit(Xs[idx], y[idx])
        counts += (np.abs(las.coef_) > 1e-8).astype(int)
    return pd.Series(counts / n_repeats, index=X.columns, name="lasso_stability")


def boruta_light(X: pd.DataFrame, y: np.ndarray, n_iter: int = 30,
                 n_estimators: int = 100, seed: int = 0) -> pd.DataFrame:
    """Lightweight Boruta: compare each feature's RF importance to max importance among shadow (shuffled) features.

    A feature is 'confirmed' if it beats the shadow max in > 50% of iterations.
    """
    rng = np.random.default_rng(seed)
    hits = np.zeros(X.shape[1], dtype=int)
    cols = list(X.columns)
    for it in range(n_iter):
        Xs = X.copy()
        # add shadow features = shuffled copies
        shadow = pd.DataFrame({f"shadow_{c}": rng.permutation(X[c].values) for c in cols},
                              index=X.index)
        Xfull = pd.concat([Xs, shadow], axis=1)
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed + it,
                                   max_depth=8, min_samples_leaf=10)
        rf.fit(Xfull.values, y)
        imps = rf.feature_importances_
        real = imps[: len(cols)]
        shad_max = imps[len(cols):].max()
        hits += (real > shad_max).astype(int)
    rate = hits / n_iter
    return pd.DataFrame({"boruta_hit_rate": rate,
                         "confirmed": rate > 0.5}, index=cols)
