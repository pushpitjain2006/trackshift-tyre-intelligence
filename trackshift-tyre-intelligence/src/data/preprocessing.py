"""
Data cleaning and preprocessing pipeline.

Rigorous filtering of invalid/non-representative laps with full
audit trail — every removed lap is tracked with a reason.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingSummary:
    """Audit trail for the preprocessing pipeline."""
    raw_count: int = 0
    valid_count: int = 0
    removed_count: int = 0
    removal_reasons: dict[str, int] = field(default_factory=dict)

    def add_removal(self, reason: str, count: int) -> None:
        self.removal_reasons[reason] = self.removal_reasons.get(reason, 0) + count
        self.removed_count += count

    def __str__(self) -> str:
        lines = [
            f"Raw laps:   {self.raw_count}",
            f"Valid laps: {self.valid_count}",
            f"Removed:    {self.removed_count}",
            "Reasons:",
        ]
        for reason, count in sorted(self.removal_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  - {reason}: {count}")
        return "\n".join(lines)


def clean_laps(
    laps_df: pd.DataFrame,
    config: Config | None = None,
) -> tuple[pd.DataFrame, PreprocessingSummary]:
    """Apply sequential cleaning filters to raw lap data.

    Parameters
    ----------
    laps_df : pd.DataFrame
        Raw laps from FastF1 or equivalent source.
    config : Config, optional
        Configuration for outlier thresholds.

    Returns
    -------
    tuple[pd.DataFrame, PreprocessingSummary]
        Cleaned laps and audit summary.
    """
    if config is None:
        config = get_config()

    summary = PreprocessingSummary(raw_count=len(laps_df))
    df = laps_df.copy()

    # Ensure LapTime_sec column exists
    if "LapTime_sec" not in df.columns and "LapTime" in df.columns:
        df["LapTime_sec"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

    # 1. Remove laps with missing lap times
    mask = df["LapTime_sec"].isna() | (df["LapTime_sec"] <= 0)
    n_removed = mask.sum()
    if n_removed > 0:
        summary.add_removal("Missing/invalid lap time", int(n_removed))
        df = df[~mask]

    # 2. Remove laps with missing driver
    if "Driver" in df.columns:
        mask = df["Driver"].isna()
        n_removed = mask.sum()
        if n_removed > 0:
            summary.add_removal("Missing driver", int(n_removed))
            df = df[~mask]

    # 3. Remove pit-in and pit-out laps
    for col, label in [("PitInTime", "Pit-in lap"), ("PitOutTime", "Pit-out lap")]:
        if col in df.columns:
            mask = df[col].notna()
            n_removed = mask.sum()
            if n_removed > 0:
                summary.add_removal(label, int(n_removed))
                df = df[~mask]

    # 4. Remove safety-car and VSC affected laps
    if "TrackStatus" in df.columns:
        # TrackStatus codes: 1=Green, 2=Yellow, 4=SC, 5=Red, 6=VSC, 7=VSC Ending
        sc_mask = df["TrackStatus"].astype(str).isin(["4", "5", "6", "7"])
        n_removed = sc_mask.sum()
        if n_removed > 0:
            summary.add_removal("Safety Car / VSC / Red Flag", int(n_removed))
            df = df[~sc_mask]

    # 5. Remove laps marked not accurate by FastF1
    if "IsAccurate" in df.columns:
        mask = df["IsAccurate"] == False  # noqa: E712
        n_removed = mask.sum()
        if n_removed > 0:
            summary.add_removal("Not accurate (FastF1 flag)", int(n_removed))
            df = df[~mask]

    # 6. Statistical outlier detection (IQR method, per driver)
    if len(df) > 0 and "Driver" in df.columns:
        outlier_mask = pd.Series(False, index=df.index)
        for driver, grp in df.groupby("Driver"):
            times = grp["LapTime_sec"]
            q1 = times.quantile(0.25)
            q3 = times.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - config.outlier.iqr_multiplier * iqr
            upper = q3 + config.outlier.iqr_multiplier * iqr
            drv_outlier = (times < lower) | (times > upper)
            outlier_mask.loc[drv_outlier.index] = drv_outlier
        n_removed = outlier_mask.sum()
        if n_removed > 0:
            summary.add_removal("Statistical outlier (IQR)", int(n_removed))
            df = df[~outlier_mask]

    # 7. Remove first lap of race (formation/grid effects)
    if "LapNumber" in df.columns:
        mask = df["LapNumber"] == 1
        n_removed = mask.sum()
        if n_removed > 0:
            summary.add_removal("First lap (formation effects)", int(n_removed))
            df = df[~mask]

    summary.valid_count = len(df)
    logger.info("Preprocessing complete:\n%s", summary)
    return df.reset_index(drop=True), summary
