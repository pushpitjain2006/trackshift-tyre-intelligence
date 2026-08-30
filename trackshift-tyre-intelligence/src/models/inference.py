"""
Model inference orchestrator with fallback hierarchy.

Runs the chosen model and automatically falls back through the hierarchy
if the primary model fails or produces unhealthy diagnostics:

    Bayesian State-Space (MCMC)
        → Fast State-Space (Kalman)
            → Multivariate Regression

The app NEVER crashes because a model fails.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import Config, get_config
from src.data import cache
from src.models.baseline import BaselineResult, fit_multivariate, fit_naive
from src.models.diagnostics import check_diagnostics
from src.models.state_space import BayesianResult

logger = logging.getLogger(__name__)


def run_degradation_analysis(
    stint_df: pd.DataFrame,
    config: Config | None = None,
    model_choice: str = "bayesian",
    use_cache: bool = True,
    cache_key_str: str = "",
) -> tuple[BayesianResult, BaselineResult, BaselineResult, str]:
    """Run the complete degradation analysis pipeline.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Fully featured stint DataFrame.
    config : Config, optional
    model_choice : str
        One of "bayesian", "kalman", "regression".
    use_cache : bool
        Whether to use cached results.
    cache_key_str : str
        Cache key for storing/retrieving results.

    Returns
    -------
    tuple[BayesianResult, BaselineResult, BaselineResult, str]
        (state_space_result, naive_result, multivariate_result, model_used)
    """
    if config is None:
        config = get_config()

    # 1. Always fit baselines (fast)
    naive = fit_naive(stint_df)
    multi = fit_multivariate(stint_df)

    # 2. Check cache for state-space result
    if use_cache and cache_key_str:
        cached = cache.load_model_result(cache_key_str)
        if cached is not None and isinstance(cached, BayesianResult):
            logger.info("Using cached state-space result.")
            return cached, naive, multi, cached.inference_mode

    # 3. Run state-space model with fallback hierarchy
    ss_result, model_used = _run_with_fallback(stint_df, config, model_choice)

    # 4. Cache the result
    if cache_key_str and ss_result is not None:
        # Don't cache the ArviZ idata (too large / not picklable cleanly)
        cache_result = BayesianResult(
            model_name=ss_result.model_name,
            degradation_mean=ss_result.degradation_mean,
            degradation_median=ss_result.degradation_median,
            degradation_p05=ss_result.degradation_p05,
            degradation_p95=ss_result.degradation_p95,
            rate_per_lap=ss_result.rate_per_lap,
            total_degradation=ss_result.total_degradation,
            tyre_age=ss_result.tyre_age,
            coefficients=ss_result.coefficients,
            diagnostics=ss_result.diagnostics,
            sampling_time_sec=ss_result.sampling_time_sec,
            inference_mode=ss_result.inference_mode,
        )
        cache.save_model_result(cache_result, cache_key_str)

    return ss_result, naive, multi, model_used


def _run_with_fallback(
    stint_df: pd.DataFrame,
    config: Config,
    model_choice: str,
) -> tuple[BayesianResult, str]:
    """Execute model with automatic fallback hierarchy."""

    # Attempt 1: Bayesian MCMC
    if model_choice in ("bayesian", "auto"):
        try:
            from src.models.state_space import build_and_fit
            logger.info("Running Bayesian state-space model (MCMC)...")
            result = build_and_fit(stint_df, config)

            # Check diagnostics
            diag = check_diagnostics(result.diagnostics)
            if diag.is_healthy:
                return result, "Bayesian State-Space (MCMC)"
            else:
                logger.warning(
                    "Bayesian model unhealthy: %s — falling back to Kalman.",
                    diag.warnings,
                )
                # Fall through to Kalman
        except Exception as exc:
            logger.warning("Bayesian model failed: %s — falling back to Kalman.", exc)

    # Attempt 2: Kalman filter
    if model_choice in ("bayesian", "kalman", "auto"):
        try:
            from src.models.fast_fallback import fit_kalman
            logger.info("Running fast Kalman state-space model...")
            result = fit_kalman(stint_df, config)
            return result, "Fast State-Space (Kalman)"
        except Exception as exc:
            logger.warning("Kalman model failed: %s — falling back to regression.", exc)

    # Attempt 3: Convert multivariate regression to BayesianResult format
    logger.info("Using multivariate regression as final fallback.")
    multi = fit_multivariate(stint_df)
    tyre_age = stint_df["tyre_age"].values
    deg_curve = multi.coefficients.get("tyre_age", 0.0) * (tyre_age - tyre_age[0])
    residual_std = multi.residuals.std()

    result = BayesianResult(
        model_name="Multivariate Regression (fallback)",
        degradation_mean=deg_curve,
        degradation_median=deg_curve,
        degradation_p05=deg_curve - 1.645 * residual_std,
        degradation_p95=deg_curve + 1.645 * residual_std,
        rate_per_lap=multi.degradation_rate_per_lap,
        total_degradation=multi.total_degradation,
        tyre_age=tyre_age,
        coefficients=multi.coefficients,
        diagnostics={"method": "OLS regression (fallback)"},
        sampling_time_sec=0.0,
        inference_mode="Multivariate Regression (fallback)",
    )
    return result, "Multivariate Regression (fallback)"
