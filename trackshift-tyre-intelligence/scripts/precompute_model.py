#!/usr/bin/env python3
"""
Precompute model results for instant demo playback.

Usage:
    python scripts/precompute_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data.demo_data import load_demo_dataset
from src.data.cache import cache_key, config_hash, save_model_result
from src.models.inference import run_degradation_analysis


def main():
    print("=" * 60)
    print("  TrackShift — Precompute Model Results")
    print("=" * 60)

    config = get_config()
    df = load_demo_dataset()

    if df is None:
        print("✗ No demo dataset found. Run build_demo_dataset.py first.")
        return

    print(f"\n→ Running models on {len(df)} laps...")

    ck = cache_key("DEMO", "SYN", 0, config_hash(config))

    bayesian_result, naive, multi, model_used = run_degradation_analysis(
        df, config, model_choice="bayesian", use_cache=False, cache_key_str=ck,
    )

    print(f"\n✓ Model: {model_used}")
    print(f"  Degradation rate: {bayesian_result.rate_per_lap:+.4f} s/lap")
    print(f"  Total degradation: {bayesian_result.total_degradation:+.3f} s")
    print(f"  Sampling time: {bayesian_result.sampling_time_sec:.1f} s")
    print(f"  Cached to: models/cached/{ck}.pkl")

    # Also precompute Kalman result
    ck_kalman = cache_key("DEMO_KALMAN", "SYN", 0, config_hash(config))
    kalman_result, _, _, kalman_model = run_degradation_analysis(
        df, config, model_choice="kalman", use_cache=False, cache_key_str=ck_kalman,
    )
    print(f"\n✓ Kalman fallback: {kalman_model}")
    print(f"  Degradation rate: {kalman_result.rate_per_lap:+.4f} s/lap")
    print(f"  Sampling time: {kalman_result.sampling_time_sec:.1f} s")

    print("\n✓ Precomputation complete!")


if __name__ == "__main__":
    main()
