"""Tests for state-space models (Kalman fallback only — MCMC too slow for CI)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.data.demo_data import generate_synthetic_stint
from src.models.fast_fallback import fit_kalman
from src.models.state_space import BayesianResult
from src.config import get_config


class TestKalmanFallback:
    def test_runs_without_error(self):
        df = generate_synthetic_stint(seed=42, num_laps=20)
        result = fit_kalman(df)
        assert isinstance(result, BayesianResult)

    def test_output_shape(self):
        df = generate_synthetic_stint(seed=42, num_laps=20)
        result = fit_kalman(df)
        n = len(df)
        assert len(result.degradation_mean) == n
        assert len(result.degradation_p05) == n
        assert len(result.degradation_p95) == n
        assert len(result.tyre_age) == n

    def test_degradation_normalized(self):
        """Degradation at lap 1 should be 0."""
        df = generate_synthetic_stint(seed=42, num_laps=20)
        result = fit_kalman(df)
        assert abs(result.degradation_mean[0]) < 1e-6

    def test_degradation_positive_trend(self):
        """On synthetic data with known degradation, trend should be positive."""
        df = generate_synthetic_stint(seed=42, num_laps=20)
        result = fit_kalman(df)
        # Last value should be greater than first (positive degradation)
        assert result.degradation_mean[-1] > result.degradation_mean[0]
        assert result.rate_per_lap > 0

    def test_uncertainty_bounds(self):
        """p05 should be below mean, p95 above."""
        df = generate_synthetic_stint(seed=42, num_laps=20)
        result = fit_kalman(df)
        assert (result.degradation_p05 <= result.degradation_mean + 1e-6).all()
        assert (result.degradation_p95 >= result.degradation_mean - 1e-6).all()

    def test_fast_execution(self):
        """Kalman should complete in < 2 seconds."""
        import time
        df = generate_synthetic_stint(seed=42, num_laps=30)
        t0 = time.time()
        fit_kalman(df)
        elapsed = time.time() - t0
        assert elapsed < 2.0

    def test_coefficients_present(self):
        df = generate_synthetic_stint(seed=42, num_laps=20)
        result = fit_kalman(df)
        assert "delta" in result.coefficients
        assert "beta0" in result.coefficients


class TestBayesianResultSchema:
    def test_default_initialization(self):
        result = BayesianResult()
        assert result.rate_per_lap == 0.0
        assert result.total_degradation == 0.0
        assert result.inference_mode == "MCMC (NUTS)"
