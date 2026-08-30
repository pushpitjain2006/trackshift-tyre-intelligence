"""
Bayesian state-space model for latent tyre degradation estimation.

The core insight: tyre degradation is a HIDDEN STATE that cannot be
directly observed. Raw lap time confounds degradation with fuel burn,
temperature, and track evolution.

Observation model:
    y_t = β₀ + β_f·f_t + β_c·c_t + β_p·p_t + α_t + ε_t

State transition:
    α_t = α_{t-1} + δ + ω_t

Where:
    y_t = observed lap time (seconds)
    f_t = estimated fuel mass (kg)
    c_t = track temperature (°C)
    p_t = track/session progression (0–1)
    α_t = latent tyre degradation state
    δ   = degradation drift (constrained ≥ 0)
    ε_t = observation noise (Student-T for robustness)
    ω_t = process noise
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    """Container for Bayesian state-space model outputs."""
    model_name: str = "Bayesian State-Space Model"
    degradation_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    degradation_median: np.ndarray = field(default_factory=lambda: np.array([]))
    degradation_p05: np.ndarray = field(default_factory=lambda: np.array([]))
    degradation_p95: np.ndarray = field(default_factory=lambda: np.array([]))
    rate_per_lap: float = 0.0
    total_degradation: float = 0.0
    tyre_age: np.ndarray = field(default_factory=lambda: np.array([]))
    coefficients: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    sampling_time_sec: float = 0.0
    inference_mode: str = "MCMC (NUTS)"
    idata: object = None  # ArviZ InferenceData (not serialized)

    @property
    def is_healthy(self) -> bool:
        """Check if sampling diagnostics are acceptable."""
        diag = self.diagnostics
        if diag.get("divergences", 0) > 10:
            return False
        if diag.get("max_rhat", 1.0) > 1.05:
            return False
        return True


def build_and_fit(
    stint_df: pd.DataFrame,
    config: Config | None = None,
) -> BayesianResult:
    """Build and fit the Bayesian state-space model.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Fully featured stint DataFrame with columns:
        ``LapTime_sec``, ``tyre_age``, ``fuel_mass_kg``,
        ``track_temp_C``, ``track_progress``.
    config : Config, optional

    Returns
    -------
    BayesianResult
        Posterior degradation curve with uncertainty.

    Raises
    ------
    RuntimeError
        If PyMC sampling fails entirely.
    """
    if config is None:
        config = get_config()

    import pymc as pm

    y = stint_df["LapTime_sec"].values.astype(float)
    n = len(y)
    fuel = stint_df["fuel_mass_kg"].values.astype(float)
    temp = stint_df["track_temp_C"].values.astype(float)
    progress = stint_df["track_progress"].values.astype(float)
    tyre_age = stint_df["tyre_age"].values.astype(float)

    # Standardize features for numerical stability
    y_mean = y.mean()
    y_std = max(y.std(), 0.1)
    fuel_mean, fuel_std = fuel.mean(), max(fuel.std(), 1.0)
    temp_mean, temp_std = temp.mean(), max(temp.std(), 1.0)
    prog_mean, prog_std = progress.mean(), max(progress.std(), 0.1)

    fuel_z = (fuel - fuel_mean) / fuel_std
    temp_z = (temp - temp_mean) / temp_std
    prog_z = (progress - prog_mean) / prog_std

    logger.info("Building PyMC model: %d laps, y_mean=%.2f, y_std=%.2f", n, y_mean, y_std)

    with pm.Model() as model:
        # --- Priors ---
        # Baseline intercept
        beta0 = pm.Normal("beta0", mu=y_mean, sigma=5.0)

        # Confounder coefficients (standardized scale)
        beta_fuel = pm.Normal("beta_fuel", mu=0, sigma=2.0)
        beta_temp = pm.Normal("beta_temp", mu=0, sigma=2.0)
        beta_prog = pm.Normal("beta_prog", mu=0, sigma=2.0)

        # Degradation drift (constrained positive)
        delta = pm.HalfNormal("delta", sigma=0.15)

        # Process and observation noise
        sigma_process = pm.HalfNormal("sigma_process", sigma=0.1)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        # Student-T degrees of freedom for robust likelihood
        if config.use_student_t:
            nu = pm.Gamma("nu", alpha=3, beta=0.5)
        
        # --- Latent degradation state (random walk with drift) ---
        innovations = pm.Normal("innovations", mu=0, sigma=1, shape=n)
        alpha_values = pm.Deterministic(
            "alpha",
            pm.math.cumsum(delta + sigma_process * innovations),
        )

        # --- Observation model ---
        mu = (
            beta0
            + beta_fuel * fuel_z
            + beta_temp * temp_z
            + beta_prog * prog_z
            + alpha_values
        )

        if config.use_student_t:
            pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma_obs, observed=y)
        else:
            pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y)

    # --- Sampling ---
    t0 = time.time()
    try:
        with model:
            idata = pm.sample(
                draws=config.sampling.draws,
                tune=config.sampling.tune,
                chains=config.sampling.chains,
                target_accept=config.sampling.target_accept,
                random_seed=config.sampling.random_seed,
                progressbar=True,
                return_inferencedata=True,
            )
        sampling_time = time.time() - t0
        logger.info("MCMC sampling completed in %.1f seconds", sampling_time)

    except Exception as exc:
        raise RuntimeError(f"PyMC sampling failed: {exc}") from exc

    # --- Extract posteriors ---
    alpha_post = idata.posterior["alpha"].values  # (chains, draws, n)
    alpha_flat = alpha_post.reshape(-1, n)  # (total_samples, n)

    # Normalize: degradation relative to first lap = 0
    alpha_relative = alpha_flat - alpha_flat[:, 0:1]

    deg_mean = alpha_relative.mean(axis=0)
    deg_median = np.median(alpha_relative, axis=0)
    deg_p05 = np.percentile(alpha_relative, 5, axis=0)
    deg_p95 = np.percentile(alpha_relative, 95, axis=0)

    # Degradation rate: mean per-lap increment
    rate = float(np.mean(np.diff(deg_mean)))
    total_deg = float(deg_mean[-1])

    # Extract coefficient posteriors
    coefficients = {}
    for var_name in ["beta0", "beta_fuel", "beta_temp", "beta_prog", "delta"]:
        if var_name in idata.posterior:
            vals = idata.posterior[var_name].values.flatten()
            coefficients[var_name] = float(np.mean(vals))

    # Un-standardize fuel/temp/prog coefficients for interpretability
    if "beta_fuel" in coefficients:
        coefficients["beta_fuel_per_kg"] = coefficients["beta_fuel"] / fuel_std
    if "beta_temp" in coefficients:
        coefficients["beta_temp_per_C"] = coefficients["beta_temp"] / temp_std

    # --- Diagnostics ---
    diagnostics = _extract_diagnostics(idata)

    result = BayesianResult(
        degradation_mean=deg_mean,
        degradation_median=deg_median,
        degradation_p05=deg_p05,
        degradation_p95=deg_p95,
        rate_per_lap=rate,
        total_degradation=total_deg,
        tyre_age=tyre_age,
        coefficients=coefficients,
        diagnostics=diagnostics,
        sampling_time_sec=sampling_time,
        idata=idata,
    )

    logger.info(
        "Bayesian result: rate=%.4f s/lap, total=%.3f s, healthy=%s",
        rate, total_deg, result.is_healthy,
    )
    return result


def _extract_diagnostics(idata) -> dict:
    """Extract MCMC diagnostics from ArviZ InferenceData."""
    diag: dict = {}
    try:
        import arviz as az

        summary = az.summary(idata, var_names=["beta0", "delta", "sigma_obs"])
        diag["max_rhat"] = float(summary["r_hat"].max()) if "r_hat" in summary.columns else 1.0
        diag["min_ess"] = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else 0
        
        # Divergences
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            diag["divergences"] = int(idata.sample_stats["diverging"].values.sum())
        else:
            diag["divergences"] = 0

        diag["summary_table"] = summary.to_dict()

    except Exception as exc:
        logger.warning("Diagnostics extraction failed: %s", exc)
        diag["max_rhat"] = 1.0
        diag["min_ess"] = 0
        diag["divergences"] = 0

    return diag
