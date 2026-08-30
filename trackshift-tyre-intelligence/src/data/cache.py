"""
Local data cache management.

Provides save/load for processed DataFrames and model results,
keyed deterministically by race + driver + stint + config hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


def cache_key(race: str, driver: str, stint: int, config_hash: str = "") -> str:
    """Generate a deterministic cache key string."""
    raw = f"{race}|{driver}|{stint}|{config_hash}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def config_hash(config: Config) -> str:
    """Generate a short hash representing current config state."""
    parts = [
        str(config.fuel.initial_fuel_kg),
        str(config.fuel.fuel_burn_per_lap_kg),
        str(config.sampling.draws),
        str(config.sampling.chains),
        str(config.use_student_t),
    ]
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def save_processed(df: pd.DataFrame, key: str, directory: Optional[Path] = None) -> Path:
    """Save a processed DataFrame as parquet."""
    if directory is None:
        from src.config import PROCESSED_DIR
        directory = PROCESSED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved processed data: %s", path)
    return path


def load_processed(key: str, directory: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load a processed DataFrame from cache."""
    if directory is None:
        from src.config import PROCESSED_DIR
        directory = PROCESSED_DIR
    path = directory / f"{key}.parquet"
    if path.exists():
        logger.info("Loaded cached data: %s", path)
        return pd.read_parquet(path)
    return None


def save_model_result(result: Any, key: str, directory: Optional[Path] = None) -> Path:
    """Pickle a model result dict."""
    if directory is None:
        from src.config import MODELS_DIR
        directory = MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved model result: %s", path)
    return path


def load_model_result(key: str, directory: Optional[Path] = None) -> Optional[Any]:
    """Load a cached model result."""
    if directory is None:
        from src.config import MODELS_DIR
        directory = MODELS_DIR
    path = directory / f"{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                result = pickle.load(f)
            logger.info("Loaded cached model: %s", path)
            return result
        except Exception as exc:
            logger.warning("Failed to load cached model %s: %s", path, exc)
    return None


def list_cached_demos(directory: Optional[Path] = None) -> list[str]:
    """List available cached demo dataset keys."""
    if directory is None:
        from src.config import DEMO_DIR
        directory = DEMO_DIR
    if not directory.exists():
        return []
    return [p.stem for p in directory.glob("*.parquet")]
