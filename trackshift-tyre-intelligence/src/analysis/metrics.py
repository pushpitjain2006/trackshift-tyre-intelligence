"""
Validation metrics for model comparison.

Provides MAE, RMSE, and temporal train/test validation.
Temporal order is always respected — no random shuffling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for model validation metrics."""
    mae: float
    rmse: float
    correlation: float
    n_train: int = 0
    n_test: int = 0
    train_mae: float = 0.0
    test_mae: float = 0.0
    train_rmse: float = 0.0
    test_rmse: float = 0.0


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> ValidationMetrics:
    """Compute MAE, RMSE, and correlation between actual and predicted values."""
    residuals = actual - predicted
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    if len(actual) > 2:
        corr = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        corr = 0.0

    return ValidationMetrics(mae=mae, rmse=rmse, correlation=corr)


def temporal_validation(
    stint_df: pd.DataFrame,
    model_fn,
    train_frac: float = 0.7,
) -> ValidationMetrics:
    """Temporal train/test split validation.

    Trains on early laps, tests on later laps — respects time ordering.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Full featured stint data.
    model_fn : callable
        Function that takes a DataFrame and returns predictions array.
    train_frac : float
        Fraction of data for training.

    Returns
    -------
    ValidationMetrics
        With train and test metrics separated.
    """
    n = len(stint_df)
    split_idx = int(n * train_frac)
    if split_idx < 5 or (n - split_idx) < 3:
        logger.warning("Insufficient data for temporal validation (%d laps).", n)
        # Fall back to full-data metrics
        preds = model_fn(stint_df)
        return compute_metrics(stint_df["LapTime_sec"].values, preds)

    train_df = stint_df.iloc[:split_idx].copy()
    test_df = stint_df.iloc[split_idx:].copy()

    try:
        train_preds = model_fn(train_df)
        test_preds = model_fn(test_df)

        train_actual = train_df["LapTime_sec"].values
        test_actual = test_df["LapTime_sec"].values

        train_metrics = compute_metrics(train_actual, train_preds)
        test_metrics = compute_metrics(test_actual, test_preds)

        return ValidationMetrics(
            mae=test_metrics.mae,
            rmse=test_metrics.rmse,
            correlation=test_metrics.correlation,
            n_train=len(train_df),
            n_test=len(test_df),
            train_mae=train_metrics.mae,
            test_mae=test_metrics.mae,
            train_rmse=train_metrics.rmse,
            test_rmse=test_metrics.rmse,
        )
    except Exception as exc:
        logger.warning("Temporal validation failed: %s", exc)
        return ValidationMetrics(mae=0, rmse=0, correlation=0)
