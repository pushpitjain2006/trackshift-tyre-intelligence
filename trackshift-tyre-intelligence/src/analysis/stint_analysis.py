"""
Stint quality scoring and automatic demo selection.

Scores stints by suitability for the demo: length, data completeness,
degradation signal visibility, and absence of anomalies.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.features.tyre_features import StintInfo

logger = logging.getLogger(__name__)


def score_stint(stint_info: StintInfo, stint_df: pd.DataFrame) -> float:
    """Score a stint for demo suitability (higher = better).

    Factors:
    - Stint length (≥15 laps preferred)
    - Valid lap ratio
    - Data completeness (weather, fuel)
    - Tyre-age range
    - Apparent degradation signal strength

    Parameters
    ----------
    stint_info : StintInfo
        Stint metadata.
    stint_df : pd.DataFrame
        Stint data with features.

    Returns
    -------
    float
        Quality score (0–100).
    """
    score = 0.0

    # Length bonus: prefer 15–35 laps
    n = stint_info.num_laps
    if n >= 15:
        score += 25.0
    elif n >= 10:
        score += 15.0
    elif n >= 8:
        score += 5.0
    else:
        return 0.0  # too short

    # Data completeness
    required = ["LapTime_sec", "tyre_age", "fuel_mass_kg"]
    completeness = sum(
        1 for col in required
        if col in stint_df.columns and stint_df[col].notna().all()
    ) / len(required)
    score += 20.0 * completeness

    # Temperature data available
    if "track_temp_C" in stint_df.columns and stint_df["track_temp_C"].notna().any():
        score += 10.0

    # Apparent degradation signal: positive correlation of residuals with tyre age
    # after a simple fuel correction
    if "LapTime_sec" in stint_df.columns and "fuel_mass_kg" in stint_df.columns:
        try:
            times = stint_df["LapTime_sec"].values
            fuel = stint_df["fuel_mass_kg"].values
            # Simple fuel correction
            fuel_effect = np.polyfit(fuel, times, 1)[0] * (fuel - fuel.mean())
            corrected = times - fuel_effect
            tyre_age = stint_df["tyre_age"].values
            corr = np.corrcoef(tyre_age, corrected)[0, 1]
            if corr > 0.3:
                score += 25.0  # strong degradation signal
            elif corr > 0.1:
                score += 15.0
            elif corr > 0:
                score += 5.0
        except Exception:
            pass

    # Lap time consistency (low variance = clean stint)
    if "LapTime_sec" in stint_df.columns:
        cv = stint_df["LapTime_sec"].std() / stint_df["LapTime_sec"].mean()
        if cv < 0.02:  # very consistent
            score += 20.0
        elif cv < 0.05:
            score += 10.0

    return score


def select_best_stint(
    all_stints: list[tuple[StintInfo, pd.DataFrame]],
) -> Optional[tuple[StintInfo, pd.DataFrame, float]]:
    """Select the highest-quality stint from a list.

    Parameters
    ----------
    all_stints : list[tuple[StintInfo, pd.DataFrame]]
        List of (stint_info, stint_df) pairs.

    Returns
    -------
    tuple[StintInfo, pd.DataFrame, float] or None
        Best (stint_info, stint_df, score), or None if no suitable stint.
    """
    if not all_stints:
        return None

    scored = []
    for info, df in all_stints:
        s = score_stint(info, df)
        scored.append((info, df, s))
        logger.info("Stint %s: score=%.1f", info.label, s)

    scored.sort(key=lambda x: -x[2])
    best = scored[0]

    if best[2] < 10.0:
        logger.warning("No high-quality stint found. Best score: %.1f", best[2])
        return None

    logger.info("Selected best stint: %s (score=%.1f)", best[0].label, best[2])
    return best
