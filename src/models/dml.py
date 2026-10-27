"""Double Machine Learning estimator — Chernozhukov et al. (2018, EctJ).

Partially Linear Regression (PLR):
    Y_i = θ · D_i + g(X_i) + ε_i          (outcome equation)
    D_i = m(X_i)  + V_i                     (first stage / nuisance)

Algorithm:
    1. K-fold cross-fitting: estimate g and m on auxiliary folds,
       compute residuals ẽ_Y and ẽ_D on held-out folds.
    2. θ̂ = Σ(ẽ_D · ẽ_Y) / Σ(ẽ_D²)         (Neyman-orthogonal moment)
    3. Sandwich SE via heteroskedasticity-robust formula.
    4. Repeat n_reps times with different fold splits → median aggregation.

Supported ML models: 'lasso', 'ridge', 'rf' (random forest), 'xgb' (XGBoost).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

MLType = Literal["lasso", "ridge", "rf", "xgb", "gbm"]


# ── ML model factory ─────────────────────────────────────────────────────────

def _make_model(name: MLType) -> object:
    """Return a sklearn-compatible estimator for the given name."""
    if name == "lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LassoCV(cv=5, max_iter=5000, n_jobs=-1)),
        ])
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RidgeCV(cv=5)),
        ])
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            n_jobs=-1, random_state=0,
        )
    if name in ("xgb", "gbm"):
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbosity=0, random_state=0,
            )
        except ImportError:
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=0,
            )
    raise ValueError(f"Unknown ml model: {name!r}")


# ── Cross-fitting helper ──────────────────────────────────────────────────────

def make_blocked_splits(dates: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-aware (expanding-window) cross-fitting splits keyed on calendar period.

    Unlike random K-fold, every training observation strictly *precedes* its
    validation observations: for each contiguous block of months used as the
    validation fold, the training set is all observations dated earlier than the
    block's first month. This removes look-ahead leakage that random shuffling
    would otherwise allow in autocorrelated monthly panel data.

    The earliest block has no prior history and is therefore *not* used as a
    validation fold (those rows receive no out-of-fold residual). With K folds
    this leaves roughly (K-1)/K of the sample with valid residuals — acceptable
    for a robustness check; the effective n is reported by the caller.

    Parameters
    ----------
    dates : (n,) array of comparable date values (one per row, may repeat
            across countries within a month).
    n_folds : number of contiguous month blocks.

    Returns
    -------
    list of (train_idx, val_idx) positional-index arrays.
    """
    dates = np.asarray(dates)
    uniq = np.array(sorted(pd.unique(dates)))
    blocks = np.array_split(uniq, n_folds)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        if len(block) == 0:
            continue
        block_min = block.min()
        val_idx   = np.where(np.isin(dates, block))[0]
        train_idx = np.where(dates < block_min)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue   # earliest block: no prior history -> skip as validation
        splits.append((train_idx, val_idx))
    if len(splits) == 0:
        raise ValueError("make_blocked_splits produced no valid time-ordered folds.")
    return splits


def _cross_fit(Y: np.ndarray, X: np.ndarray,
               ml_model, n_folds: int, seed: int,
               shuffle: bool = True,
               splits: list[tuple[np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
    """K-fold cross-fitting: return residuals Y - Ê[Y|X].

    Estimate E[Y|X] on each training fold; predict on held-out fold.
    No data leakage between folds.

    splits is not None -> use the supplied (train_idx, val_idx) folds verbatim
                          (used for blocked time-aware cross-fitting). Rows that
                          never appear in a validation fold keep a NaN residual.
    shuffle=True        -> conventional random K-fold (baseline).
    shuffle=False       -> contiguous (non-shuffled) K-fold.
    """
    n = len(Y)
    residuals = np.full(n, np.nan)

    if splits is None:
        kf = KFold(n_splits=n_folds, shuffle=shuffle,
                   random_state=seed if shuffle else None)
        splits = list(kf.split(X))

    for train_idx, test_idx in splits:
        model = clone(ml_model)
        model.fit(X[train_idx], Y[train_idx])
        residuals[test_idx] = Y[test_idx] - model.predict(X[test_idx])

    return residuals


# ── DML result container ─────────────────────────────────────────────────────

@dataclass
class DMLResult:
    theta:    float
    se:       float
    t_stat:   float
    p_value:  float
    ci_lower: float
    ci_upper: float
    n_obs:    int
    treatment: str = ""
    outcome:   str = ""
    country:   str = ""
    ml_model:  str = ""
    n_folds:   int = 5
    n_reps:    int = 3

    def summary(self) -> str:
        sig = "***" if self.p_value < 0.01 else \
              "**"  if self.p_value < 0.05 else \
              "*"   if self.p_value < 0.10 else ""
        return (
            f"DML — {self.country} | Y={self.outcome} | D={self.treatment}\n"
            f"  θ = {self.theta:+.4f}  SE = {self.se:.4f}  "
            f"t = {self.t_stat:+.3f}  p = {self.p_value:.4f} {sig}\n"
            f"  95% CI = [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  "
            f"n = {self.n_obs}  ML = {self.ml_model}"
        )

    def to_dict(self) -> dict:
        return {
            "country":   self.country,
            "outcome":   self.outcome,
            "treatment": self.treatment,
            "theta":     round(self.theta, 6),
            "se":        round(self.se, 6),
            "t_stat":    round(self.t_stat, 4),
            "p_value":   round(self.p_value, 4),
            "ci_lower":  round(self.ci_lower, 6),
            "ci_upper":  round(self.ci_upper, 6),
            "n_obs":     self.n_obs,
            "ml_model":  self.ml_model,
            "n_folds":   self.n_folds,
            "n_reps":    self.n_reps,
            "signif":    "***" if self.p_value < 0.01 else
                         "**"  if self.p_value < 0.05 else
                         "*"   if self.p_value < 0.10 else "",
        }


# ── Main DML estimator ────────────────────────────────────────────────────────

class DMLEstimator:
    """Double Machine Learning for the Partially Linear Regression model.

    Parameters
    ----------
    ml_g : str or sklearn estimator
        ML method to estimate g(X) = E[Y|X].  str shortcuts: 'lasso','ridge','rf','xgb'.
    ml_m : str or sklearn estimator
        ML method to estimate m(X) = E[D|X].  Same shortcuts as ml_g.
        If None, uses the same spec as ml_g.
    n_folds : int
        Number of cross-fitting folds (K in the paper).  Recommend 5.
    n_reps : int
        Number of repetitions with different random seeds.
        Final θ is the median across reps (reduces sensitivity to fold draws).
    seed : int
        Base random seed.
    """

    def __init__(
        self,
        ml_g: MLType | object = "lasso",
        ml_m: MLType | object | None = None,
        n_folds: int = 5,
        n_reps:  int = 3,
        seed:    int = 42,
        cv_scheme: Literal["shuffled", "blocked"] = "shuffled",
    ) -> None:
        self.ml_g    = ml_g
        self.ml_m    = ml_m if ml_m is not None else ml_g
        self.n_folds = n_folds
        self.n_reps  = n_reps
        self.seed    = seed
        # "shuffled" = conventional random K-fold (baseline);
        # "blocked"  = time-aware contiguous folds (robustness).
        self.cv_scheme = cv_scheme

        self._ml_g_obj = _make_model(ml_g) if isinstance(ml_g, str) else ml_g
        self._ml_m_obj = _make_model(self.ml_m) if isinstance(self.ml_m, str) else self.ml_m
        self._ml_name  = ml_g if isinstance(ml_g, str) else type(ml_g).__name__

    def fit(
        self,
        Y: np.ndarray | pd.Series,
        D: np.ndarray | pd.Series,
        X: np.ndarray | pd.DataFrame,
        *,
        treatment: str = "D",
        outcome:   str = "Y",
        country:   str = "",
        dates:     np.ndarray | pd.Series | None = None,
    ) -> DMLResult:
        """Estimate θ via DML cross-fitting.

        Parameters
        ----------
        Y : (n,) array  — outcome variable
        D : (n,) array  — treatment variable (one column)
        X : (n, p) array — confounders / nuisance controls
        dates : (n,) array of period labels, required when cv_scheme="blocked"
                so that time-aware (expanding-window) folds can be built.

        Returns
        -------
        DMLResult with theta, se, t_stat, p_value, 95% CI
        """
        Y = np.asarray(Y, dtype=float)
        D = np.asarray(D, dtype=float)
        X = np.asarray(X, dtype=float)
        dates_arr = None if dates is None else np.asarray(dates)

        # Drop rows with any NaN across Y, D, X
        mask = ~(np.isnan(Y) | np.isnan(D) | np.isnan(X).any(axis=1))
        Y, D, X = Y[mask], D[mask], X[mask]
        if dates_arr is not None:
            dates_arr = dates_arr[mask]
        n = len(Y)

        if n < 30:
            raise ValueError(
                f"Too few observations after NaN removal: {n}. "
                "Need at least 30 for reliable DML inference."
            )

        # ── Cross-fitting: n_reps repetitions ────────────────────────────────
        shuffle = (self.cv_scheme == "shuffled")
        # Blocked (time-aware) folds are deterministic, so a single pass suffices.
        n_reps_eff = self.n_reps if shuffle else 1

        blocked_splits = None
        if self.cv_scheme == "blocked":
            if dates_arr is None:
                raise ValueError("cv_scheme='blocked' requires `dates` so that "
                                 "time-ordered folds can be constructed.")
            blocked_splits = make_blocked_splits(dates_arr, self.n_folds)

        theta_reps: list[float] = []
        eY_all = np.full((n, n_reps_eff), np.nan)
        eD_all = np.full((n, n_reps_eff), np.nan)

        for r in range(n_reps_eff):
            seed_r = self.seed + r * 100
            eY = _cross_fit(Y, X, self._ml_g_obj, self.n_folds, seed_r,
                            shuffle=shuffle, splits=blocked_splits)
            eD = _cross_fit(D, X, self._ml_m_obj, self.n_folds, seed_r,
                            shuffle=shuffle, splits=blocked_splits)

            # Neyman-orthogonal theta estimate for this rep (drop NaN rows that
            # blocked folds leave un-residualised).
            v = ~(np.isnan(eY) | np.isnan(eD))
            denom = np.dot(eD[v], eD[v])
            if denom < 1e-12:
                raise ValueError("First-stage residuals ẽ_D are nearly zero. "
                                  "D may be perfectly predicted by X.")
            theta_r = np.dot(eD[v], eY[v]) / denom
            theta_reps.append(theta_r)
            eY_all[:, r] = eY
            eD_all[:, r] = eD

        # Aggregate across reps (median for robustness)
        theta = float(np.median(theta_reps))

        # ── Standard error: sandwich formula using averaged residuals ─────────
        # Use mean residuals across reps; restrict to rows with valid residuals.
        eY_avg = np.nanmean(eY_all, axis=1)
        eD_avg = np.nanmean(eD_all, axis=1)
        valid  = ~(np.isnan(eY_avg) | np.isnan(eD_avg))
        eY_avg, eD_avg = eY_avg[valid], eD_avg[valid]
        n_eff  = int(valid.sum())
        eps    = eY_avg - theta * eD_avg          # structural residual

        # Sandwich: SE² = E[ẽ_D² · ε²] / (E[ẽ_D²])²  × (1/n)
        score_sq = (eD_avg * eps) ** 2
        denom_sq = (np.mean(eD_avg ** 2)) ** 2
        var_theta = np.mean(score_sq) / (n_eff * denom_sq)
        se = float(np.sqrt(max(var_theta, 0.0)))

        if se < 1e-12:
            se = 1e-8  # prevent division by zero in edge cases

        # ── Inference ─────────────────────────────────────────────────────────
        t_stat  = theta / se
        p_value = float(2 * (1 - norm.cdf(abs(t_stat))))
        ci_lo   = theta - 1.96 * se
        ci_hi   = theta + 1.96 * se

        return DMLResult(
            theta=theta, se=se, t_stat=t_stat, p_value=p_value,
            ci_lower=ci_lo, ci_upper=ci_hi,
            n_obs=n_eff, treatment=treatment, outcome=outcome,
            country=country, ml_model=self._ml_name,
            n_folds=self.n_folds, n_reps=n_reps_eff,
        )

    def fit_panel(
        self,
        panel: pd.DataFrame,
        outcome_col:   str,
        treatment_col: str,
        control_cols:  list[str],
        country_col:   str = "country",
        *,
        pooled: bool = False,
    ) -> pd.DataFrame:
        """Run DML for each country in the panel.

        Parameters
        ----------
        panel : DataFrame with country column and time index
        outcome_col : column name for Y
        treatment_col : column name for D
        control_cols : list of column names for X
        pooled : if True, also run pooled (all countries) regression

        Returns
        -------
        DataFrame of DMLResult.to_dict() for each country (+ 'ALL' if pooled)
        """
        results: list[dict] = []
        countries = panel[country_col].unique() if country_col in panel.columns else ["ALL"]

        for cc in countries:
            sub = panel[panel[country_col] == cc].copy() if country_col in panel.columns \
                  else panel.copy()
            sub = sub.dropna(subset=[outcome_col, treatment_col] + control_cols)
            if len(sub) < 30:
                print(f"[DML] {cc}: only {len(sub)} obs — skipping")
                continue
            try:
                res = self.fit(
                    Y=sub[outcome_col].values,
                    D=sub[treatment_col].values,
                    X=sub[control_cols].values,
                    treatment=treatment_col,
                    outcome=outcome_col,
                    country=cc,
                )
                print(res.summary())
                results.append(res.to_dict())
            except Exception as e:
                print(f"[DML] {cc} failed: {e}")

        if pooled and country_col in panel.columns:
            sub = panel.dropna(subset=[outcome_col, treatment_col] + control_cols)
            # Add country dummies as extra controls
            dummies = pd.get_dummies(sub[country_col], prefix="cty", drop_first=True)
            X_pool  = pd.concat([sub[control_cols].reset_index(drop=True),
                                  dummies.reset_index(drop=True)], axis=1).values
            try:
                res = self.fit(
                    Y=sub[outcome_col].values,
                    D=sub[treatment_col].values,
                    X=X_pool,
                    treatment=treatment_col,
                    outcome=outcome_col,
                    country="POOLED",
                )
                print(res.summary())
                results.append(res.to_dict())
            except Exception as e:
                print(f"[DML] POOLED failed: {e}")

        return pd.DataFrame(results)


# ── Convenience wrapper for sub-period analysis ───────────────────────────────

def dml_subperiods(
    df: pd.DataFrame,
    outcome_col:   str,
    treatment_col: str,
    control_cols:  list[str],
    periods: dict[str, tuple[str, str]],   # {"pre_covid": ("2000-01", "2020-01"), ...}
    ml_model: MLType = "lasso",
    country:  str = "",
) -> pd.DataFrame:
    """Run DML for each sub-period and return comparison table."""
    estimator = DMLEstimator(ml_g=ml_model, n_folds=5, n_reps=3)
    rows: list[dict] = []

    for label, (t_start, t_end) in periods.items():
        sub = df.loc[t_start:t_end].dropna(
            subset=[outcome_col, treatment_col] + control_cols
        )
        if len(sub) < 30:
            print(f"[DML] sub-period {label}: only {len(sub)} obs — skipping")
            continue
        try:
            res = estimator.fit(
                Y=sub[outcome_col].values,
                D=sub[treatment_col].values,
                X=sub[control_cols].values,
                treatment=treatment_col,
                outcome=outcome_col,
                country=f"{country}_{label}",
            )
            d = res.to_dict()
            d["period"] = label
            d["t_start"] = t_start
            d["t_end"]   = t_end
            rows.append(d)
            print(res.summary())
        except Exception as e:
            print(f"[DML] sub-period {label} failed: {e}")

    return pd.DataFrame(rows)
