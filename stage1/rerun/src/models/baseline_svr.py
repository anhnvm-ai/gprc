"""SVR baseline (sklearn) — works on flattened (lookback × features) windows."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

@dataclass
class SVRBaseline:
    C_grid: list = None
    eps_grid: list = None
    gamma_grid: list = None

    def fit(self, X3d: np.ndarray, y: np.ndarray):
        Xf = X3d.reshape(X3d.shape[0], -1)
        pipe = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf"))])
        param = {
            "svr__C":       self.C_grid    or [0.1, 1, 10, 100],
            "svr__epsilon": self.eps_grid  or [0.001, 0.01, 0.1],
            "svr__gamma":   self.gamma_grid or ["scale", 0.01, 0.1],
        }
        gs = GridSearchCV(pipe, param, cv=TimeSeriesSplit(5),
                          scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(Xf, y)
        self.model_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        return self

    def predict(self, X3d: np.ndarray) -> np.ndarray:
        return self.model_.predict(X3d.reshape(X3d.shape[0], -1))
