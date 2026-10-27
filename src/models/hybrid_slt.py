"""Hybrid SVR -> LSTM -> Transformer with constrained Ridge meta-learner.

Pipeline (formal spec — see README §6):
    Stage 1: SVR on flattened windows -> base prediction r_hat_svr; residuals e1.
    Stage 2: LSTM on [X, r_hat_svr_broadcast] -> predicts e1; export hidden states H.
    Stage 3: Transformer encoder on [H || interaction channel] -> predicts e2.
    Meta:    Constrained Ridge stacks the three predictions (non-neg, sum-to-one).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
from src.models.baseline_svr import SVRBaseline
from src.models.baseline_lstm import LSTMBaseline
from src.models.baseline_transformer import TransformerBaseline

def _broadcast_pred(pred_1d: np.ndarray, lookback: int) -> np.ndarray:
    """Broadcast a (N,) prediction along the lookback dimension as an extra channel."""
    return np.repeat(pred_1d[:, None, None], lookback, axis=1)   # (N, L, 1)

def _constrained_ridge(P: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """Solve min ||P w - y||^2 + l2||w||^2  s.t. w >= 0, sum(w)=1."""
    K = P.shape[1]
    def obj(w):
        e = P @ w - y
        return float(e @ e) + l2 * float(w @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * K
    w0 = np.ones(K) / K
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
    return res.x

@dataclass
class HybridSLT:
    lookback: int = 22
    svr_kwargs: dict = field(default_factory=dict)
    lstm_kwargs: dict = field(default_factory=dict)
    tfm_kwargs: dict = field(default_factory=dict)
    meta: str = "ridge"   # 'ridge' (constrained) or 'lgbm'
    lgbm_kwargs: dict = field(default_factory=dict)

    def fit(self, X_tr, y_tr, X_va, y_va, interaction_tr=None, interaction_va=None):
        # --- Stage 1: SVR ---
        self.svr_ = SVRBaseline(**self.svr_kwargs).fit(X_tr, y_tr)
        s_tr = self.svr_.predict(X_tr); s_va = self.svr_.predict(X_va)
        e1_tr = y_tr - s_tr; e1_va = y_va - s_va

        # --- Stage 2: LSTM on [X, broadcast(s)] predicting residual e1 ---
        X_tr_aug = np.concatenate([X_tr, _broadcast_pred(s_tr, self.lookback)], axis=2)
        X_va_aug = np.concatenate([X_va, _broadcast_pred(s_va, self.lookback)], axis=2)
        self.lstm_ = LSTMBaseline(**self.lstm_kwargs).fit(X_tr_aug, e1_tr, X_va_aug, e1_va)
        l_tr = self.lstm_.predict(X_tr_aug); l_va = self.lstm_.predict(X_va_aug)
        r2_tr = s_tr + l_tr; r2_va = s_va + l_va
        e2_tr = y_tr - r2_tr; e2_va = y_va - r2_va

        # --- Stage 3: Transformer on hidden states (+ interaction channel) predicting e2 ---
        H_tr = self.lstm_.hidden_states(X_tr_aug)
        H_va = self.lstm_.hidden_states(X_va_aug)
        if interaction_tr is not None:
            ch_tr = _broadcast_pred(np.asarray(interaction_tr, dtype=float), H_tr.shape[1])
            ch_va = _broadcast_pred(np.asarray(interaction_va, dtype=float), H_va.shape[1])
            H_tr = np.concatenate([H_tr, ch_tr], axis=2)
            H_va = np.concatenate([H_va, ch_va], axis=2)
        self.tfm_ = TransformerBaseline(**self.tfm_kwargs).fit(H_tr, e2_tr, H_va, e2_va)
        t_tr = self.tfm_.predict(H_tr); t_va = self.tfm_.predict(H_va)
        r3_va = r2_va + t_va

        # --- Meta-learner: stack (s_va, r2_va, r3_va) ---
        P_va = np.stack([s_va, r2_va, r3_va], axis=1)
        if self.meta == "lgbm":
            from lightgbm import LGBMRegressor
            params = dict(n_estimators=200, learning_rate=0.03, num_leaves=15,
                          min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                          random_state=0, verbose=-1)
            params.update(self.lgbm_kwargs)
            self.meta_model_ = LGBMRegressor(**params).fit(P_va, y_va)
            self.weights_ = np.array([np.nan, np.nan, np.nan])  # not linear
        else:
            self.weights_ = _constrained_ridge(P_va, y_va)
        return self

    def predict(self, X, interaction=None):
        s = self.svr_.predict(X)
        X_aug = np.concatenate([X, _broadcast_pred(s, self.lookback)], axis=2)
        l = self.lstm_.predict(X_aug); r2 = s + l
        H = self.lstm_.hidden_states(X_aug)
        if interaction is not None:
            ch = _broadcast_pred(np.asarray(interaction, dtype=float), H.shape[1])
            H = np.concatenate([H, ch], axis=2)
        t = self.tfm_.predict(H); r3 = r2 + t
        P = np.stack([s, r2, r3], axis=1)
        if self.meta == "lgbm":
            return self.meta_model_.predict(P)
        return P @ self.weights_
