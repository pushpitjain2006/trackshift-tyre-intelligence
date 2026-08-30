#!/usr/bin/env python3
"""
Build the demo dataset — either from FastF1 or synthetic fallback.

Usage:
    python scripts/build_demo_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data.demo_data import generate_synthetic_stint, save_demo_dataset
from src.ev.translation import generate_mock_ev_data


def main():
    print("=" * 60)
    print("  TrackShift — Build Demo Dataset")
    print("=" * 60)

    # 1. Try FastF1 first
    config = get_config()
    fastf1_success = False

    try:
        from src.data.fastf1_loader import load_session
        from src.data.preprocessing import clean_laps
        from src.features.tyre_features import extract_stints, get_stint_df
        from src.features.confounders import build_feature_matrix
        from src.analysis.stint_analysis import select_best_stint, score_stint

        print(f"\n→ Loading {config.race.year} {config.race.grand_prix}...")
        laps_df, weather_df, error = load_session(
            config.race.year, config.race.grand_prix, config.race.session_type,
            cache_dir=str(config.raw_dir),
        )

        if laps_df is not None:
            print(f"  ✓ Loaded {len(laps_df)} raw laps")

            clean_df, summary = clean_laps(laps_df, config)
            print(f"  ✓ {summary.valid_count} valid laps after cleaning")

            # Find best stint across all drivers
            from src.data.fastf1_loader import list_drivers
            drivers = list_drivers(clean_df)
            all_stints = []

            for driver in drivers:
                stints = extract_stints(clean_df, driver, config)
                for stint_info in stints:
                    stint_df = get_stint_df(clean_df, stint_info)
                    stint_df = build_feature_matrix(stint_df, weather_df, config)
                    all_stints.append((stint_info, stint_df))

            best = select_best_stint(all_stints)
            if best:
                info, df, score = best
                print(f"  ✓ Best stint: {info.label} (score={score:.1f})")
                df["is_synthetic"] = False
                save_demo_dataset(df, "demo_stint")
                fastf1_success = True
        else:
            print(f"  ✗ FastF1 failed: {error}")

    except Exception as exc:
        print(f"  ✗ FastF1 pipeline failed: {exc}")

    # 2. Always generate synthetic as backup
    print("\n→ Generating synthetic demo dataset...")
    synthetic = generate_synthetic_stint()
    save_demo_dataset(synthetic, "synthetic_stint")
    print(f"  ✓ {len(synthetic)} synthetic laps saved")

    if not fastf1_success:
        # Use synthetic as primary demo
        save_demo_dataset(synthetic, "demo_stint")
        print("  ℹ Using synthetic data as primary demo (FastF1 unavailable)")

    # 3. Generate mock EV data
    print("\n→ Generating mock EV telemetry...")
    ev_data = generate_mock_ev_data()
    ev_path = config.demo_dir / "mock_ev_telemetry.csv"
    ev_data.to_csv(ev_path, index=False)
    print(f"  ✓ {len(ev_data)} EV trips saved to {ev_path}")

    print("\n✓ Demo dataset build complete!")


if __name__ == "__main__":
    main()
