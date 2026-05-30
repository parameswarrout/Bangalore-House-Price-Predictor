"""Shared regression metrics for training and evaluation."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true_log, y_pred_log) -> dict:
    """Score models trained on log1p(price); report log R² and lakhs-scale errors."""
    y_true = np.expm1(y_true_log)
    y_hat = np.expm1(y_pred_log)
    abs_err = np.abs(y_true - y_hat)
    mape = float(np.mean(abs_err / np.maximum(y_true, 1e-6)) * 100)
    return {
        "r2": round(float(r2_score(y_true_log, y_pred_log)), 4),
        "mae_lakhs": round(float(mean_absolute_error(y_true, y_hat)), 2),
        "rmse_lakhs": round(float(np.sqrt(mean_squared_error(y_true, y_hat))), 2),
        "mape_pct": round(mape, 2),
        "median_ae_lakhs": round(float(np.median(abs_err)), 2),
    }
