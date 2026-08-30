"""
Central configuration for TrackShift Tyre Degradation Intelligence.

All configurable parameters live here — fuel assumptions, sampling budget,
paths, default race selection. Nothing is invisibly hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = _PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = _PROJECT_ROOT / "models" / "cached"

# Ensure directories exist
for _d in (RAW_DIR, PROCESSED_DIR, DEMO_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# F1 constants
# ---------------------------------------------------------------------------
@dataclass
class FuelConfig:
    """Fuel-mass estimation parameters.

    These are *estimates* — exact race fuel loads are not publicly observable.
    """
    initial_fuel_kg: float = 110.0
    fuel_burn_per_lap_kg: float = 1.8  # ~1.5–2.0 kg/lap typical range


@dataclass
class SamplingConfig:
    """PyMC MCMC sampling budget."""
    draws: int = 500
    tune: int = 500
    chains: int = 2
    target_accept: float = 0.9
    random_seed: int = 42


@dataclass
class RaceConfig:
    """Default race for demo mode."""
    year: int = 2024
    grand_prix: str = "Monza"
    session_type: str = "R"  # Race


@dataclass
class OutlierConfig:
    """Statistical outlier detection thresholds."""
    iqr_multiplier: float = 1.5
    min_stint_laps: int = 8  # minimum laps for a usable stint


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    fuel: FuelConfig = field(default_factory=FuelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    race: RaceConfig = field(default_factory=RaceConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)

    # Paths
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    demo_dir: Path = DEMO_DIR
    models_dir: Path = MODELS_DIR

    # Feature flags
    use_student_t: bool = True  # robust observation noise
    enable_track_evolution: bool = True
    enable_temperature: bool = True


def get_config() -> Config:
    """Return the default configuration."""
    return Config()
