"""
conformal.py — Split (inductive) conformal prediction for RUL regression,
plus the coverage/efficiency diagnostics most RUL papers never report.

Two nonconformity scores are implemented:
  1. Absolute residual (standard split conformal): symmetric, fixed-width
     intervals. Simple, but ignores that RUL uncertainty is *heteroscedastic*
     (a battery at cycle 5 is much harder to forecast to failure than one
     at cycle 160 near end-of-life).
  2. Normalized residual: nonconformity is |y - yhat| / sigma_hat, where
     sigma_hat is a difficulty proxy (here: predicted residual magnitude
     from a secondary model). This gives locally-adaptive interval widths
     and is the version worth reporting as the main result — the width vs.
     cycle-number plot is a good sanity-check figure for the paper.

Guarantee: for exchangeable calibration/test data, P(y in interval) >= 1 - alpha.
NASA battery data across DIFFERENT cells is not strictly exchangeable with a
held-out cell (distribution shift), so this script explicitly measures and
reports empirical coverage on the held-out battery rather than assuming the
guarantee transfers — that gap IS the interesting empirical finding.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def split_conformal_intervals(y_cal, yhat_cal, y_test, yhat_test, alpha=0.1):
    """Standard (symmetric) split conformal. Returns lower, upper, q_hat."""
    n = len(y_cal)
    scores = np.abs(y_cal - yhat_cal)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(scores, q_level, method="higher")
    lower = yhat_test - q_hat
    upper = yhat_test + q_hat
    return lower, upper, q_hat


def normalized_conformal_intervals(X_cal, y_cal, yhat_cal, X_test, yhat_test, alpha=0.1, seed=0):
    """
    Locally-adaptive split conformal. A secondary RF is trained to predict
    |residual| from X_cal (a difficulty/heteroscedasticity model), used to
    scale the nonconformity score.
    """
    resid_cal = np.abs(y_cal - yhat_cal)
    difficulty_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=seed, n_jobs=-1)
    difficulty_model.fit(X_cal, resid_cal)

    sigma_cal = np.clip(difficulty_model.predict(X_cal), 1e-3, None)
    scores = resid_cal / sigma_cal

    n = len(y_cal)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(scores, q_level, method="higher")

    sigma_test = np.clip(difficulty_model.predict(X_test), 1e-3, None)
    lower = yhat_test - q_hat * sigma_test
    upper = yhat_test + q_hat * sigma_test
    return lower, upper, q_hat, difficulty_model


def empirical_coverage(y_true, lower, upper):
    covered = (y_true >= lower) & (y_true <= upper)
    return covered.mean()


def mean_interval_width(lower, upper):
    return np.mean(upper - lower)
