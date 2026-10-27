"""Forecast metrics."""
from __future__ import annotations
import numpy as np

def rmse(y, yh): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yh)) ** 2)))
def mae(y, yh):  return float(np.mean(np.abs(np.asarray(y) - np.asarray(yh))))
def mape(y, yh, eps=1e-8):
    y, yh = np.asarray(y), np.asarray(yh)
    return float(np.mean(np.abs((y - yh) / (np.abs(y) + eps))) * 100)
def directional_accuracy(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    return float(np.mean(np.sign(y) == np.sign(yh)))
def qlike(sig2_true, sig2_hat, eps=1e-12):
    """QLIKE for variance forecasts (Patton 2011). Inputs are variances (>0)."""
    s, h = np.asarray(sig2_true), np.maximum(np.asarray(sig2_hat), eps)
    return float(np.mean(s / h - np.log(s / h) - 1))

def all_metrics(y, yh, kind: str = "return") -> dict:
    out = {"RMSE": rmse(y, yh), "MAE": mae(y, yh), "MAPE": mape(y, yh)}
    if kind == "return":
        out["DA"] = directional_accuracy(y, yh)
    elif kind == "vol":
        out["QLIKE"] = qlike(np.maximum(y, 1e-8) ** 2, np.maximum(yh, 1e-8) ** 2)
    return out
