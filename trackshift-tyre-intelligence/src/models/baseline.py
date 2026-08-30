"""
Baseline regression models for tyre degradation.

Two models that demonstrate progressively better understanding:

1. Naive: LapTime ~ TyreAge  (ignores confounders — intentionally misleading)
2. Multivariate: LapTime ~ TyreAge + Fuel + Temp + TrackProgress  (corrected)

The purpose is to show that naive analysis produces wrong conclusions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result container for baseline regression models."""
    model_name: str
    coefficients: dict[str, float]
    intercept: float
    predictions: np.ndarray
    residuals: np.ndarray
    r_squared: float
    feature_names: list[str]
    degradation_rate_per_lap: float  # estimated from tyre_age coefficient
    total_degradation: float  # over the stint

    @property
    def summary(self) -> str:
        lines = [f"=== {self.model_name} ==="]
        lines.append(f"R² = {self.r_squared:.4f}")
        lines.append(f"Intercept = {self.intercept:.3f}")
        for name, coef in self.coefficients.items():
            lines.append(f"  {name}: {coef:+.4f}")
        lines.append(f"Degradation rate: {self.degradation_rate_per_lap:+.4f} s/lap")
        lines.append(f"Total degradation: {self.total_degradation:+.3f} s")
        return "\n".join(lines)


def fit_naive(stint_df: pd.DataFrame) -> BaselineResult:
    """Fit naive regression: LapTime = α + β·TyreAge + ε.

    This model intentionally ignores fuel, temperature, and track evolution.
    It will typically produce a *negative* or near-zero degradation coefficient
    because fuel burn makes lap times decrease, masking tyre degradation.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Must contain ``tyre_age`` and ``LapTime_sec``.

    Returns
    -------
    BaselineResult
    """
    X = stint_df[["tyre_age"]].values
    y = stint_df["LapTime_sec"].values

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions

    tyre_coef = float(model.coef_[0])
    n_laps = len(stint_df)

    result = BaselineResult(
        model_name="Naive Regression (TyreAge only)",
        coefficients={"tyre_age": tyre_coef},
        intercept=float(model.intercept_),
        predictions=predictions,
        residuals=residuals,
        r_squared=float(model.score(X, y)),
        feature_names=["tyre_age"],
        degradation_rate_per_lap=tyre_coef,
        total_degradation=tyre_coef * n_laps,
    )

    logger.info("Naive model: %s", result.summary)
    return result


def fit_multivariate(stint_df: pd.DataFrame) -> BaselineResult:
    """Fit confounder-aware regression:

    LapTime = α + β_tyre·TyreAge + β_fuel·Fuel + β_temp·Temp + β_prog·Progress + ε

    This model accounts for the major confounders, producing a more
    accurate (typically positive) degradation coefficient.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Must contain ``tyre_age``, ``fuel_mass_kg``, ``track_temp_C``,
        ``track_progress``, and ``LapTime_sec``.

    Returns
    -------
    BaselineResult
    """
    feature_cols = ["tyre_age", "fuel_mass_kg", "track_temp_C", "track_progress"]
    available = [c for c in feature_cols if c in stint_df.columns]

    if not available:
        raise ValueError("No feature columns available for multivariate regression.")

    X = stint_df[available].values
    y = stint_df["LapTime_sec"].values

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions

    coefficients = {name: float(coef) for name, coef in zip(available, model.coef_)}
    tyre_coef = coefficients.get("tyre_age", 0.0)
    n_laps = len(stint_df)

    result = BaselineResult(
        model_name="Multivariate Regression (confounder-aware)",
        coefficients=coefficients,
        intercept=float(model.intercept_),
        predictions=predictions,
        residuals=residuals,
        r_squared=float(model.score(X, y)),
        feature_names=available,
        degradation_rate_per_lap=tyre_coef,
        total_degradation=tyre_coef * n_laps,
    )

    logger.info("Multivariate model: %s", result.summary)
    return result
