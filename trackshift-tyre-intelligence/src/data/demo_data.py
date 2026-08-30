"""
Synthetic demo data generator and offline demo dataset loader.

Generates realistic F1-like tyre degradation data with known ground truth
for model validation AND provides a reliable offline fallback when FastF1
is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, get_config, DEMO_DIR

logger = logging.getLogger(__name__)


def generate_synthetic_stint(
    seed: int = 42,
    config: Config | None = None,
    num_laps: int = 28,
    base_lap_time: float = 82.0,
    compound: str = "MEDIUM",
) -> pd.DataFrame:
    """Generate a single synthetic tyre stint with known ground truth.

    The generated data demonstrates the core paradox: raw lap times can
    *decrease* (car gets faster) even as tyres degrade, because fuel
    burn reduces car mass faster than tyres lose grip.

    Ground truth degradation follows:
        true_degradation = base_rate × tyre_age + nonlinear_term

    Observed lap time:
        lap_time = base + degradation + fuel_effect + temp_effect + track_effect + noise

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    config : Config, optional
    num_laps : int
        Number of laps in the stint.
    base_lap_time : float
        Baseline lap time in seconds.
    compound : str
        Tyre compound name.

    Returns
    -------
    pd.DataFrame
        Columns: LapNumber, tyre_age, fuel_mass_kg, track_temp_C, air_temp_C,
        track_progress, true_degradation, LapTime_sec, Driver, Compound,
        is_synthetic.
    """
    if config is None:
        config = get_config()

    rng = np.random.RandomState(seed)

    # --- Ground truth parameters ---
    degradation_rate = 0.055  # seconds/lap base degradation
    nonlinear_coeff = 0.0015  # quadratic degradation component
    fuel_time_effect = -0.035  # seconds per kg of fuel (lighter = faster)
    temp_effect = 0.015  # seconds per °C above baseline
    track_evolution_effect = -0.3  # total track improvement over race
    temp_baseline = 35.0  # reference track temperature

    # Starting lap in the race (stint 2 starts around lap 18)
    start_lap = 18
    lap_numbers = np.arange(start_lap, start_lap + num_laps)
    tyre_age = np.arange(1, num_laps + 1, dtype=float)

    # Fuel mass (decreasing linearly from race start)
    fuel_mass = np.maximum(
        config.fuel.initial_fuel_kg - config.fuel.fuel_burn_per_lap_kg * (lap_numbers - 1),
        0.0,
    )

    # Track temperature (slight variation through stint)
    track_temp = temp_baseline + 3.0 * np.sin(np.linspace(0, np.pi, num_laps)) + rng.normal(0, 0.5, num_laps)
    air_temp = track_temp - 8.0 + rng.normal(0, 0.3, num_laps)

    # Track progress (normalized 0-1 over full race)
    total_race_laps = 53  # typical Monza race
    track_progress = lap_numbers / total_race_laps

    # --- Ground truth degradation ---
    true_degradation = degradation_rate * tyre_age + nonlinear_coeff * tyre_age ** 2

    # --- Component effects ---
    fuel_effect = fuel_time_effect * (fuel_mass - config.fuel.initial_fuel_kg)
    temp_component = temp_effect * (track_temp - temp_baseline)
    track_component = track_evolution_effect * track_progress

    # --- Noise: slightly right-skewed (driver errors add time) ---
    base_noise = rng.normal(0, 0.15, num_laps)
    # Add occasional positive outliers (lockups, traffic, mistakes)
    outlier_mask = rng.random(num_laps) < 0.08  # ~8% of laps
    base_noise[outlier_mask] += rng.exponential(0.8, outlier_mask.sum())

    # --- Observed lap time ---
    lap_times = (
        base_lap_time
        + true_degradation
        + fuel_effect
        + temp_component
        + track_component
        + base_noise
    )

    df = pd.DataFrame({
        "LapNumber": lap_numbers,
        "tyre_age": tyre_age,
        "fuel_mass_kg": fuel_mass,
        "track_temp_C": track_temp,
        "air_temp_C": air_temp,
        "track_progress": track_progress,
        "true_degradation": true_degradation,
        "fuel_effect": fuel_effect,
        "temp_effect": temp_component,
        "track_effect": track_component,
        "LapTime_sec": lap_times,
        "Driver": "SYN",
        "Compound": compound,
        "Stint": 2,
        "is_synthetic": True,
    })

    logger.info(
        "Generated synthetic stint: %d laps, true deg rate=%.3f s/lap, "
        "total true deg=%.2f s",
        num_laps,
        degradation_rate,
        true_degradation[-1],
    )
    return df


def save_demo_dataset(df: pd.DataFrame, name: str = "demo_stint") -> Path:
    """Save a demo dataset to the demo directory."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    path = DEMO_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved demo dataset: %s", path)
    return path


def load_demo_dataset(name: str = "demo_stint") -> pd.DataFrame | None:
    """Load a bundled demo dataset.

    If no pre-built demo exists, generates a fresh synthetic one.
    """
    path = DEMO_DIR / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        logger.info("Loaded demo dataset from %s (%d laps)", path, len(df))
        return df

    # Generate on-the-fly if not pre-built
    logger.info("No cached demo found — generating synthetic dataset.")
    df = generate_synthetic_stint()
    save_demo_dataset(df, name)
    return df


def generate_demo_metadata() -> dict:
    """Return metadata for the synthetic demo stint."""
    return {
        "race": "Synthetic Demo (based on Monza profile)",
        "driver": "SYN",
        "compound": "MEDIUM",
        "stint_number": 2,
        "data_source": "SYNTHETIC VALIDATION DATASET",
        "note": (
            "This is deterministic synthetic data with known ground truth. "
            "It is NOT real F1 data. Use it for model validation and as an "
            "offline fallback."
        ),
    }
