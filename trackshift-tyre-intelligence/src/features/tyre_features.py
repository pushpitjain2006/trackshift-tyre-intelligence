"""
Tyre stint extraction and tyre-age calculation.

A stint is continuous running on the same compound after a tyre change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class StintInfo:
    """Metadata for a single tyre stint."""
    driver: str
    compound: str
    stint_number: int
    start_lap: int
    end_lap: int
    num_laps: int
    mean_lap_time: float
    compound_color: str = ""

    @property
    def label(self) -> str:
        return f"Stint {self.stint_number}: {self.compound} (Laps {self.start_lap}-{self.end_lap})"


# Compound colors following F1 convention
COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC107",
    "HARD": "#EEEEEE",
    "INTERMEDIATE": "#43A047",
    "WET": "#1E88E5",
}


def extract_stints(
    clean_laps_df: pd.DataFrame,
    driver: str,
    config: Config | None = None,
) -> list[StintInfo]:
    """Extract tyre stints for a single driver.

    Parameters
    ----------
    clean_laps_df : pd.DataFrame
        Cleaned lap data (post-preprocessing).
    driver : str
        Driver abbreviation (e.g., "VER", "HAM").
    config : Config, optional

    Returns
    -------
    list[StintInfo]
        Ordered list of stint metadata objects.
    """
    if config is None:
        config = get_config()

    df = clean_laps_df[clean_laps_df["Driver"] == driver].copy()
    if len(df) == 0:
        return []

    df = df.sort_values("LapNumber").reset_index(drop=True)

    stints: list[StintInfo] = []

    # Use Stint column if available (FastF1 provides this)
    if "Stint" in df.columns:
        for stint_num, grp in df.groupby("Stint"):
            if len(grp) < config.outlier.min_stint_laps:
                continue
            compound = _get_compound(grp)
            info = StintInfo(
                driver=driver,
                compound=compound,
                stint_number=int(stint_num),
                start_lap=int(grp["LapNumber"].min()),
                end_lap=int(grp["LapNumber"].max()),
                num_laps=len(grp),
                mean_lap_time=float(grp["LapTime_sec"].mean()),
                compound_color=COMPOUND_COLORS.get(compound, "#AAAAAA"),
            )
            stints.append(info)
    else:
        # Fallback: detect stint boundaries from compound changes
        if "Compound" not in df.columns:
            # If no compound data, treat entire run as one stint
            stints.append(StintInfo(
                driver=driver,
                compound="UNKNOWN",
                stint_number=1,
                start_lap=int(df["LapNumber"].min()),
                end_lap=int(df["LapNumber"].max()),
                num_laps=len(df),
                mean_lap_time=float(df["LapTime_sec"].mean()),
            ))
        else:
            stint_num = 1
            compound_changes = df["Compound"].ne(df["Compound"].shift()).cumsum()
            for _, grp in df.groupby(compound_changes):
                if len(grp) < config.outlier.min_stint_laps:
                    continue
                compound = _get_compound(grp)
                info = StintInfo(
                    driver=driver,
                    compound=compound,
                    stint_number=stint_num,
                    start_lap=int(grp["LapNumber"].min()),
                    end_lap=int(grp["LapNumber"].max()),
                    num_laps=len(grp),
                    mean_lap_time=float(grp["LapTime_sec"].mean()),
                    compound_color=COMPOUND_COLORS.get(compound, "#AAAAAA"),
                )
                stints.append(info)
                stint_num += 1

    logger.info("Driver %s: found %d usable stints", driver, len(stints))
    return stints


def get_stint_df(
    clean_laps_df: pd.DataFrame,
    stint_info: StintInfo,
) -> pd.DataFrame:
    """Extract the lap data for a specific stint and add tyre_age column.

    Parameters
    ----------
    clean_laps_df : pd.DataFrame
        Cleaned session laps.
    stint_info : StintInfo
        Stint metadata from extract_stints().

    Returns
    -------
    pd.DataFrame
        Stint laps with ``tyre_age`` column (1-indexed, first lap on tyre = 1).
    """
    df = clean_laps_df[
        (clean_laps_df["Driver"] == stint_info.driver)
        & (clean_laps_df["LapNumber"] >= stint_info.start_lap)
        & (clean_laps_df["LapNumber"] <= stint_info.end_lap)
    ].copy()

    df = df.sort_values("LapNumber").reset_index(drop=True)

    # Tyre age: use TyreLife if available, otherwise derive from position in stint
    if "TyreLife" in df.columns and df["TyreLife"].notna().all():
        df["tyre_age"] = df["TyreLife"].astype(float)
    else:
        df["tyre_age"] = np.arange(1, len(df) + 1, dtype=float)

    return df


def _get_compound(grp: pd.DataFrame) -> str:
    """Extract the most common compound label from a group of laps."""
    if "Compound" in grp.columns:
        mode = grp["Compound"].mode()
        if len(mode) > 0:
            return str(mode.iloc[0]).upper()
    return "UNKNOWN"
