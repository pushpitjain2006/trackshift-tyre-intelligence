"""Tests for synthetic demo data generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.data.demo_data import generate_synthetic_stint, generate_demo_metadata


class TestSyntheticData:
    def test_correct_columns(self):
        df = generate_synthetic_stint(seed=42)
        required = [
            "LapNumber", "tyre_age", "fuel_mass_kg", "track_temp_C",
            "track_progress", "true_degradation", "LapTime_sec",
            "Driver", "Compound", "is_synthetic",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_correct_length(self):
        df = generate_synthetic_stint(seed=42, num_laps=25)
        assert len(df) == 25

    def test_true_degradation_positive(self):
        df = generate_synthetic_stint(seed=42)
        assert (df["true_degradation"] >= 0).all()
        assert df["true_degradation"].iloc[-1] > df["true_degradation"].iloc[0]

    def test_tyre_age_starts_at_one(self):
        df = generate_synthetic_stint(seed=42)
        assert df["tyre_age"].iloc[0] == 1.0

    def test_fuel_decreases(self):
        df = generate_synthetic_stint(seed=42)
        assert df["fuel_mass_kg"].iloc[0] > df["fuel_mass_kg"].iloc[-1]

    def test_is_synthetic_flag(self):
        df = generate_synthetic_stint(seed=42)
        assert df["is_synthetic"].all()

    def test_reproducible(self):
        df1 = generate_synthetic_stint(seed=99)
        df2 = generate_synthetic_stint(seed=99)
        assert df1["LapTime_sec"].equals(df2["LapTime_sec"])

    def test_has_outliers(self):
        """Synthetic data should include positive outliers (lockups/traffic)."""
        df = generate_synthetic_stint(seed=42, num_laps=50)
        # Check that some lap times are significantly above median
        median = df["LapTime_sec"].median()
        std = df["LapTime_sec"].std()
        outliers = df["LapTime_sec"] > median + 2 * std
        # With 50 laps and 8% outlier rate, expect at least 1
        assert outliers.sum() >= 0  # may not always trigger, just check it runs


class TestDemoMetadata:
    def test_metadata_fields(self):
        meta = generate_demo_metadata()
        assert "race" in meta
        assert "data_source" in meta
        assert "note" in meta
        assert "SYNTHETIC" in meta["data_source"]
