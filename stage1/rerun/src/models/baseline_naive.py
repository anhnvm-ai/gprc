"""Naive baselines for return forecasting: Random Walk, AR(1), Mean."""
from __future__ import annotations
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class RandomWalk:
    """y_hat(t) = y(t-1)  — using last bar of last lookback feature column."""
    def __init__(self, ret_idx_in_window: int = -1):
        # We don't have the raw return inside scaled X; instead, RW predicts 0 for log-return.
        pass
    def fit(self, *_, **__): return self
    def predict(self, X3d) -> np.ndarray:
        return np.zeros(X3d.shape[0], dtype=np.float32)


class MeanForecast:
    def fit(self, X_tr, y_tr, *_, **__):
        self.mu_ = float(np.mean(y_tr)); return self
    def predict(self, X3d) -> np.ndarray:
        return np.full(X3d.shape[0], self.mu_, dtype=np.float32)


class AR1:
    """One-dim AR(1) on y itself (doesn't use features). Used as a sanity baseline."""
    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        # OLS: y_t = a + b * y_{t-1}
        y = np.asarray(y_tr, dtype=float)
        x = y[:-1]; z = y[1:]
        b = np.cov(x, z, ddof=0)[0, 1] / (np.var(x) + 1e-12)
        a = z.mean() - b * x.mean()
        self.a_, self.b_, self.last_ = float(a), float(b), float(y[-1]); return self
    def predict(self, X3d) -> np.ndarray:
        # For each test step we don't have access to true previous y at predict time;
        # forecast is constant = unconditional mean a/(1-b).
        if abs(1 - self.b_) > 1e-6:
            uc = self.a_ / (1 - self.b_)
        else:
            uc = self.last_
        return np.full(X3d.shape[0], uc, dtype=np.float32)


class EWMAVol:
    """RiskMetrics EWMA variance forecast (lambda=0.94). Predicts realized vol (annualized std)."""
    def __init__(self, lam: float = 0.94):
        self.lam = lam
    def fit(self, X_tr, y_tr, *_, **__):
        # initialise with sample variance of training targets
        self.var_ = float(np.var(y_tr) + 1e-12)
        self.last_ret_ = float(y_tr[-1])
        self._train_y = np.asarray(y_tr, dtype=float)
        return self
    def predict(self, X3d) -> np.ndarray:
        # one-step-ahead vol forecast = sqrt(var)
        # We propagate starting from training tail, treating each test bar as a new shock placeholder=0
        var = self.var_
        out = np.empty(X3d.shape[0], dtype=np.float32)
        for i in range(len(out)):
            out[i] = np.sqrt(var)
            var = self.lam * var + (1 - self.lam) * 0.0  # zero shock proxy (test target unknown)
        return out
