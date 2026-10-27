"""GARCH(1,1)-t baseline for volatility forecasting.

Trained on a 1-D series of log-returns; produces 1-step-ahead conditional vol (annualized).
Refit periodically (every `refit_freq` bars) to avoid look-ahead.
"""
from __future__ import annotations
import numpy as np


class GARCH11t:
    def __init__(self, refit_freq: int = 21):
        self.refit_freq = refit_freq

    def fit(self, returns_train: np.ndarray):
        from arch import arch_model
        self._train = np.asarray(returns_train, dtype=float) * 100.0  # arch expects %
        am = arch_model(self._train, vol="Garch", p=1, q=1, dist="t")
        self.res_ = am.fit(disp="off")
        return self

    def forecast_path(self, returns_test: np.ndarray) -> np.ndarray:
        """Rolling 1-step-ahead conditional vol over the test sample (annualized)."""
        from arch import arch_model
        full = np.concatenate([self._train, np.asarray(returns_test, dtype=float) * 100.0])
        n_train = len(self._train); n_test = len(returns_test)
        out = np.empty(n_test, dtype=float)
        last_fit = -1
        for i in range(n_test):
            # refit every `refit_freq` bars using all data up to (n_train + i - 1)
            if (i - last_fit) >= self.refit_freq or last_fit == -1:
                hist = full[: n_train + i]
                if len(hist) >= 252:
                    res = arch_model(hist, vol="Garch", p=1, q=1, dist="t").fit(disp="off")
                    self.res_ = res; last_fit = i
            f = self.res_.forecast(horizon=1, reindex=False)
            sig = float(np.sqrt(f.variance.values[-1, 0]))
            out[i] = sig
        # back from % daily to annualized fraction
        return (out / 100.0) * np.sqrt(252)
