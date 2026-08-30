"""Tests for feature engineering: tyre age, fuel, confounders."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.config import get_config
from src.features.fuel import estimate_fuel_mass
from src.features.tyre_features import extract_stints, get_stint_df, StintInfo
from src.features.confounders import add_track_evolution, join_weather


class TestFuelEstimation:
    def test_fuel_decreases(self):
        df = pd.DataFrame({"LapNumber": [1, 10, 20, 30]})
        result = estimate_fuel_mass(df)
        assert result["fuel_mass_kg"].iloc[0] > result["fuel_mass_kg"].iloc[-1]

    def test_fuel_starts_at_initial(self):
        config = get_config()
        df = pd.DataFrame({"LapNumber": [1]})
        result = estimate_fuel_mass(df, config)
        assert result["fuel_mass_kg"].iloc[0] == config.fuel.initial_fuel_kg

    def test_fuel_never_negative(self):
        df = pd.DataFrame({"LapNumber": [1, 50, 100, 200]})
        result = estimate_fuel_mass(df)
        assert (result["fuel_mass_kg"] >= 0).all()

    def test_fuel_burn_rate(self):
        config = get_config()
        df = pd.DataFrame({"LapNumber": [1, 2]})
        result = estimate_fuel_mass(df, config)
        expected_drop = config.fuel.fuel_burn_per_lap_kg
        actual_drop = result["fuel_mass_kg"].iloc[0] - result["fuel_mass_kg"].iloc[1]
        assert abs(actual_drop - expected_drop) < 1e-6


class TestTyreAge:
    def _make_laps(self):
        return pd.DataFrame({
            "LapNumber": list(range(2, 32)),
            "LapTime_sec": np.random.normal(82, 0.3, 30),
            "Driver": ["VER"] * 15 + ["VER"] * 15,
            "Compound": ["MEDIUM"] * 15 + ["HARD"] * 15,
            "Stint": [1] * 15 + [2] * 15,
        })

    def test_stint_extraction_count(self):
        df = self._make_laps()
        stints = extract_stints(df, "VER")
        assert len(stints) == 2

    def test_stint_compounds(self):
        df = self._make_laps()
        stints = extract_stints(df, "VER")
        compounds = [s.compound for s in stints]
        assert "MEDIUM" in compounds
        assert "HARD" in compounds

    def test_tyre_age_starts_at_one(self):
        df = self._make_laps()
        stints = extract_stints(df, "VER")
        for stint in stints:
            stint_df = get_stint_df(df, stint)
            assert stint_df["tyre_age"].iloc[0] == 1.0

    def test_tyre_age_increases(self):
        df = self._make_laps()
        stints = extract_stints(df, "VER")
        for stint in stints:
            stint_df = get_stint_df(df, stint)
            assert (stint_df["tyre_age"].diff().dropna() > 0).all()


class TestTrackEvolution:
    def test_progress_bounded(self):
        df = pd.DataFrame({"LapNumber": [1, 25, 50]})
        result = add_track_evolution(df)
        assert result["track_progress"].min() >= 0
        assert result["track_progress"].max() <= 1.0

    def test_progress_monotonic(self):
        df = pd.DataFrame({"LapNumber": list(range(1, 20))})
        result = add_track_evolution(df)
        assert (result["track_progress"].diff().dropna() >= 0).all()


class TestWeatherJoining:
    def test_no_weather_returns_nan(self):
        df = pd.DataFrame({"LapNumber": [1, 2, 3]})
        result = join_weather(df, None)
        assert "track_temp_C" in result.columns
        assert result["track_temp_C"].isna().all()

    def test_weather_joined(self):
        df = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapStartDate": pd.to_datetime(["2024-09-01 14:00:00",
                                             "2024-09-01 14:01:30",
                                             "2024-09-01 14:03:00"]),
        })
        weather = pd.DataFrame({
            "Time": pd.to_datetime(["2024-09-01 13:59:00",
                                    "2024-09-01 14:01:00",
                                    "2024-09-01 14:02:30"]),
            "TrackTemp": [35.0, 36.0, 37.0],
            "AirTemp": [28.0, 29.0, 30.0],
        })
        result = join_weather(df, weather)
        assert result["track_temp_C"].notna().all()
        assert result["air_temp_C"].notna().all()
