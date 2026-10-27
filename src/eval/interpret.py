"""Interpretability: SHAP (KernelExplainer on flattened windows) + Permutation FI."""
from __future__ import annotations
import numpy as np, pandas as pd

def permutation_importance_seq(predict_fn, X3d, y, n_repeats: int = 5, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = np.mean((y - predict_fn(X3d)) ** 2)
    scores = {}
    n_feat = X3d.shape[2]
    for f in range(n_feat):
        diffs = []
        for _ in range(n_repeats):
            Xp = X3d.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, :, f] = Xp[perm, :, f]
            diffs.append(np.mean((y - predict_fn(Xp)) ** 2) - base)
        scores[f] = float(np.mean(diffs))
    return pd.Series(scores).sort_values(ascending=False)

def shap_kernel_flat(predict_fn, X3d_background, X3d_explain, feature_names: list[str] | None = None,
                     nsamples: int = 100):
    import shap
    bg = X3d_background.reshape(X3d_background.shape[0], -1)
    ex = X3d_explain.reshape(X3d_explain.shape[0], -1)
    f  = lambda Xf: predict_fn(Xf.reshape(-1, X3d_background.shape[1], X3d_background.shape[2]))
    expl = shap.KernelExplainer(f, shap.sample(bg, 50))
    sv = expl.shap_values(ex, nsamples=nsamples)
    return sv
