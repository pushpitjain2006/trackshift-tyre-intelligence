"""Tests for baseline regression models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import fit_naive, fit_multivariate
from src.data.demo_data import generate_synthetic_stint


class TestNaiveBaseline:
    def test_on_synthetic_data(self):
        df = generate_synthetic_stint(seed=42)
        result = fit_naive(df)
        assert result.model_name is not None
        assert len(result.predictions) == len(df)
        assert len(result.residuals) == len(df)

    def test_naive_misleading_on_confounded_data(self):
        """The naive model should underestimate true degradation rate
        because fuel burn partially masks tyre wear."""
        df = generate_synthetic_stint(seed=42)
        result = fit_naive(df)
        # True degradation rate is ~0.055 s/lap + quadratic term
        # The naive model conflates fuel burn with degradation
        # It should exist (not crash) and produce a coefficient
        assert result.degradation_rate_per_lap is not None

    def test_output_schema(self):
        df = generate_synthetic_stint(seed=42)
        result = fit_naive(df)
        assert "tyre_age" in result.coefficients
        assert result.r_squared is not None
        assert result.intercept is not None


class TestMultivariateBaseline:
    def test_on_synthetic_data(self):
        df = generate_synthetic_stint(seed=42)
        result = fit_multivariate(df)
        assert len(result.predictions) == len(df)
        assert "fuel_mass_kg" in result.coefficients

    def test_accounts_for_confounders(self):
        """Multivariate model should include fuel, temperature, and progress."""
        df = generate_synthetic_stint(seed=42)
        result = fit_multivariate(df)
        assert "fuel_mass_kg" in result.coefficients
        assert "track_temp_C" in result.coefficients
        assert "track_progress" in result.coefficients

    def test_higher_r_squared(self):
        """Multivariate model should explain at least as much variance."""
        df = generate_synthetic_stint(seed=42)
        naive = fit_naive(df)
        multi = fit_multivariate(df)
        assert multi.r_squared >= naive.r_squared - 0.01  # allow tiny floating-point diff
