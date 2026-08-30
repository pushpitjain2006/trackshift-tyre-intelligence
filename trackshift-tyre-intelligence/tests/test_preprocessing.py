"""Tests for data preprocessing pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import clean_laps, PreprocessingSummary
from src.config import get_config


def _make_sample_laps(n: int = 50) -> pd.DataFrame:
    """Create a sample laps DataFrame mimicking FastF1 output."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "LapNumber": np.arange(1, n + 1),
        "LapTime_sec": 82.0 + rng.normal(0, 0.3, n),
        "Driver": "VER",
        "Compound": "MEDIUM",
        "Stint": 1,
        "TrackStatus": "1",
        "IsAccurate": True,
        "TyreLife": np.arange(1, n + 1),
    })
    # Add pit in/out for lap 25
    df.loc[24, "PitInTime"] = pd.Timedelta("0:01:30")
    df.loc[25, "PitOutTime"] = pd.Timedelta("0:01:30")
    # Add NaN lap time
    df.loc[30, "LapTime_sec"] = np.nan
    # Add huge outlier
    df.loc[40, "LapTime_sec"] = 120.0
    return df


class TestCleanLaps:
    def test_removes_missing_lap_time(self):
        df = _make_sample_laps()
        clean, summary = clean_laps(df)
        assert clean["LapTime_sec"].isna().sum() == 0
        assert "Missing/invalid lap time" in summary.removal_reasons

    def test_removes_pit_laps(self):
        df = _make_sample_laps()
        clean, summary = clean_laps(df)
        reasons = summary.removal_reasons
        assert "Pit-in lap" in reasons or "Pit-out lap" in reasons

    def test_removes_first_lap(self):
        df = _make_sample_laps()
        clean, summary = clean_laps(df)
        assert 1 not in clean["LapNumber"].values
        assert "First lap (formation effects)" in summary.removal_reasons

    def test_removes_outliers(self):
        df = _make_sample_laps()
        clean, summary = clean_laps(df)
        # The 120s outlier should be removed
        assert clean["LapTime_sec"].max() < 100

    def test_summary_counts_consistent(self):
        df = _make_sample_laps()
        clean, summary = clean_laps(df)
        assert summary.raw_count == len(df)
        assert summary.valid_count == len(clean)
        assert summary.removed_count == summary.raw_count - summary.valid_count

    def test_empty_input(self):
        df = pd.DataFrame({"LapTime_sec": [], "Driver": [], "LapNumber": []})
        clean, summary = clean_laps(df)
        assert len(clean) == 0
        assert summary.raw_count == 0
