"""
FastF1 session loader with graceful failure handling.

Loads lap-level data and weather from the FastF1 API.
Telemetry is NOT loaded — lap-level data is sufficient for the prototype.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_session(
    year: int,
    grand_prix: str,
    session_type: str = "R",
    cache_dir: str = "data/raw",
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Load an F1 session via FastF1.

    Parameters
    ----------
    year : int
        Season year (e.g., 2024).
    grand_prix : str
        Grand Prix name or round number (e.g., "Monza", "Italian Grand Prix").
    session_type : str
        Session identifier — "R" for Race, "Q" for Qualifying, etc.
    cache_dir : str
        Directory for FastF1's built-in file cache.

    Returns
    -------
    tuple[laps_df | None, weather_df | None, error_msg | None]
        On success: (laps DataFrame, weather DataFrame, None)
        On failure: (None, None, error message string)
    """
    try:
        import fastf1
    except ImportError:
        return None, None, "FastF1 is not installed."

    try:
        from pathlib import Path
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_path))

        logger.info("Loading FastF1 session: %d %s %s", year, grand_prix, session_type)
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(telemetry=False, weather=True)

        laps = session.laps
        if laps is None or len(laps) == 0:
            return None, None, f"No lap data found for {year} {grand_prix} {session_type}."

        # Convert to plain pandas DataFrame
        laps_df = pd.DataFrame(laps)

        # Extract weather data
        weather_df = None
        if session.weather_data is not None and len(session.weather_data) > 0:
            weather_df = pd.DataFrame(session.weather_data)

        # Store session metadata
        laps_df.attrs["year"] = year
        laps_df.attrs["grand_prix"] = grand_prix
        laps_df.attrs["session_type"] = session_type
        laps_df.attrs["event_name"] = str(getattr(session, "event", {}).get("EventName", grand_prix))

        logger.info(
            "Loaded %d laps, weather=%s",
            len(laps_df),
            "available" if weather_df is not None else "unavailable",
        )
        return laps_df, weather_df, None

    except Exception as exc:
        error_msg = f"FastF1 loading failed: {exc}"
        logger.warning(error_msg)
        return None, None, error_msg


def list_drivers(laps_df: pd.DataFrame) -> list[str]:
    """Return sorted list of driver abbreviations in the session."""
    if "Driver" in laps_df.columns:
        return sorted(laps_df["Driver"].dropna().unique().tolist())
    return []


def list_available_races() -> list[dict]:
    """Return a curated list of recommended races for the demo.

    These are races known to have clean, long stints with observable
    degradation — suitable for demonstrating the tyre intelligence system.
    """
    return [
        {"year": 2024, "gp": "Monza", "label": "2024 Italian GP (Monza)"},
        {"year": 2024, "gp": "Spain", "label": "2024 Spanish GP (Barcelona)"},
        {"year": 2024, "gp": "Bahrain", "label": "2024 Bahrain GP (Sakhir)"},
        {"year": 2023, "gp": "Bahrain", "label": "2023 Bahrain GP (Sakhir)"},
        {"year": 2024, "gp": "Japan", "label": "2024 Japanese GP (Suzuka)"},
        {"year": 2024, "gp": "Belgium", "label": "2024 Belgian GP (Spa)"},
    ]
