"""
Confounder feature engineering — temperature joining and track evolution.

These variables change lap time independently of tyre wear.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.features.fuel import estimate_fuel_mass

logger = logging.getLogger(__name__)


def join_weather(
    stint_df: pd.DataFrame,
    weather_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join weather observations to laps by nearest timestamp.

    Uses backward-looking nearest match to avoid look-ahead leakage:
    each lap gets the most recent weather reading at or before the lap start.

    Parameters
    ----------
    stint_df : pd.DataFrame
        Must contain a time reference (``LapStartDate`` or ``Time``).
    weather_df : pd.DataFrame | None
        Weather data with ``Time``, ``TrackTemp``, ``AirTemp`` columns.

    Returns
    -------
    pd.DataFrame
        Input with ``track_temp_C`` and ``air_temp_C`` columns added.
    """
    df = stint_df.copy()

    if weather_df is None or len(weather_df) == 0:
        logger.warning("No weather data available — using fallback values.")
        df["track_temp_C"] = np.nan
        df["air_temp_C"] = np.nan
        return df

    # Determine lap timestamp column
    time_col = None
    for candidate in ["LapStartDate", "Time", "Date"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        logger.warning("No timestamp column found — cannot join weather.")
        df["track_temp_C"] = np.nan
        df["air_temp_C"] = np.nan
        return df

    try:
        # Ensure both are datetime
        lap_times = pd.to_datetime(df[time_col], errors="coerce")
        weather = weather_df.copy()
        weather["Time"] = pd.to_datetime(weather["Time"], errors="coerce")
        weather = weather.dropna(subset=["Time"]).sort_values("Time")

        track_temps = []
        air_temps = []
        for lt in lap_times:
            if pd.isna(lt):
                track_temps.append(np.nan)
                air_temps.append(np.nan)
                continue
            # Backward-looking: get most recent weather at or before lap start
            mask = weather["Time"] <= lt
            if mask.any():
                row = weather[mask].iloc[-1]
                track_temps.append(float(row.get("TrackTemp", np.nan)))
                air_temps.append(float(row.get("AirTemp", np.nan)))
            else:
                # Fallback to nearest available
                idx = (weather["Time"] - lt).abs().idxmin()
                row = weather.loc[idx]
                track_temps.append(float(row.get("TrackTemp", np.nan)))
                air_temps.append(float(row.get("AirTemp", np.nan)))

        df["track_temp_C"] = track_temps
        df["air_temp_C"] = air_temps

        logger.info(
            "Weather joined: track temp %.1f–%.1f°C",
            df["track_temp_C"].min(),
            df["track_temp_C"].max(),
        )
    except Exception as exc:
        logger.warning("Weather join failed: %s", exc)
        df["track_temp_C"] = np.nan
        df["air_temp_C"] = np.nan

    return df


def add_track_evolution(stint_df: pd.DataFrame) -> pd.DataFrame:
    """Add session/track evolution feature.

    Track evolution represents the gradual improvement in grip as rubber
    is laid down on the racing surface throughout the session.

    Feature: ``track_progress`` = normalized lap number (0 to 1).

    Parameters
    ----------
    stint_df : pd.DataFrame
        Must contain ``LapNumber`` column.

    Returns
    -------
    pd.DataFrame
        Input with ``track_progress`` column added.
    """
    df = stint_df.copy()

    if "LapNumber" not in df.columns:
        df["track_progress"] = 0.5  # neutral fallback
        return df

    max_lap = df["LapNumber"].max()
    if max_lap <= 0:
        df["track_progress"] = 0.5
    else:
        df["track_progress"] = df["LapNumber"].astype(float) / float(max_lap)

    return df


def build_feature_matrix(
    stint_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Orchestrate all confounder feature engineering.

    Adds columns:
    - ``fuel_mass_kg`` — estimated fuel mass
    - ``track_temp_C`` — track surface temperature
    - ``air_temp_C`` — air temperature
    - ``track_progress`` — normalized session progression

    Parameters
    ----------
    stint_df : pd.DataFrame
        Stint laps with ``tyre_age`` and ``LapTime_sec``.
    weather_df : pd.DataFrame | None
        Session weather data.
    config : Config, optional

    Returns
    -------
    pd.DataFrame
        Fully featured stint DataFrame.
    """
    if config is None:
        config = get_config()

    df = stint_df.copy()

    # Fuel estimation
    df = estimate_fuel_mass(df, config)

    # Weather / temperature
    if config.enable_temperature:
        df = join_weather(df, weather_df)
    else:
        df["track_temp_C"] = np.nan
        df["air_temp_C"] = np.nan

    # Track evolution
    if config.enable_track_evolution:
        df = add_track_evolution(df)
    else:
        df["track_progress"] = 0.5

    # Handle missing temperature by using stint median (no look-ahead within stint)
    for col in ["track_temp_C", "air_temp_C"]:
        if df[col].isna().all():
            df[col] = 30.0  # reasonable default
        elif df[col].isna().any():
            df[col] = df[col].ffill().bfill()  # forward-fill, then back-fill

    logger.info(
        "Feature matrix built: %d laps, columns=%s",
        len(df),
        [c for c in ["tyre_age", "fuel_mass_kg", "track_temp_C", "track_progress"] if c in df.columns],
    )
    return df
