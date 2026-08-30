"""
Fuel mass estimation.

Fuel mass is a critical confounder: as fuel burns off the car gets lighter
and faster, masking tyre degradation. Exact fuel loads are not public —
we estimate using FIA-compliant assumptions.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)


def estimate_fuel_mass(
    stint_df: pd.DataFrame,
    config: Config | None = None,
    total_race_laps: int | None = None,
) -> pd.DataFrame:
    """Add ``fuel_mass_kg`` column based on estimated linear fuel burn.

    Formula:
        fuel_mass = initial_fuel - burn_rate × (lap_number - 1)

    The lap_number is the *race* lap number (absolute), not stint-relative,
    because fuel burns from the race start.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Must contain ``LapNumber`` column.
    config : Config, optional
    total_race_laps : int, optional
        Total laps in the race. Used to clamp fuel to ≥ 0.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``fuel_mass_kg`` column added.
    """
    if config is None:
        config = get_config()

    df = stint_df.copy()

    if "LapNumber" not in df.columns:
        logger.warning("No LapNumber column — cannot estimate fuel mass.")
        df["fuel_mass_kg"] = config.fuel.initial_fuel_kg
        return df

    fuel = (
        config.fuel.initial_fuel_kg
        - config.fuel.fuel_burn_per_lap_kg * (df["LapNumber"].astype(float) - 1.0)
    )
    # Clamp to non-negative
    df["fuel_mass_kg"] = np.maximum(fuel, 0.0)

    logger.info(
        "Fuel mass estimated: %.1f kg (lap %d) → %.1f kg (lap %d)",
        df["fuel_mass_kg"].iloc[0],
        int(df["LapNumber"].iloc[0]),
        df["fuel_mass_kg"].iloc[-1],
        int(df["LapNumber"].iloc[-1]),
    )
    return df
