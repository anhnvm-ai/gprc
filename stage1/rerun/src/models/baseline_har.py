"""HAR-RV baseline for realized-volatility forecasting (Corsi, 2009).

Model:
    RV_t+1 = β0 + β1 * RV_t  +  β2 * mean(RV_{t-4..t})  +  β3 * mean(RV_{t-21..t}) + ε

Optionally extends with a single exogenous covariate (e.g., GPR) → HAR-RV-X.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class HARRV:
    def __init__(self, exog: pd.Series | None = None):
        self.exog_full = exog  # optional, indexed by full date range

    @staticmethod
    def _build_features(rv: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({
            "RV_d": rv.shift(1),
            "RV_w": rv.shift(1).rolling(5).mean(),
            "RV_m": rv.shift(1).rolling(22).mean(),
        })
        return df

    def fit(self, rv_train: pd.Series):
        X = self._build_features(rv_train)
        if self.exog_full is not None:
            X["X"] = self.exog_full.reindex(rv_train.index).shift(1)
        y = rv_train
        df = pd.concat([X, y.rename("y")], axis=1).dropna()
        Xm = np.column_stack([np.ones(len(df)), df.drop(columns="y").values])
        ym = df["y"].values
        # OLS closed form
        self.beta_, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        self.cols_ = list(df.drop(columns="y").columns)
        return self

    def predict(self, rv_full: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
        """Use *realized* RV history at every step (RV is observable from returns).
        rv_full must include both the training history AND the test period values."""
        rv_full = rv_full[~rv_full.index.duplicated(keep="first")].sort_index()
        X = self._build_features(rv_full).reindex(test_index)
        if self.exog_full is not None:
            x = self.exog_full.reindex(rv_full.index).ffill().shift(1)
            X["X"] = x.reindex(test_index)
        # any remaining NaN (e.g. first dates) → fill with column mean used in training
        X = X.fillna(method="ffill").fillna(0.0)
        Xm = np.column_stack([np.ones(len(X)), X[self.cols_].values])
        return Xm @ self.beta_
