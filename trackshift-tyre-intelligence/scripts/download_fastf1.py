#!/usr/bin/env python3
"""
Download FastF1 data for specified races and cache locally.

Usage:
    python scripts/download_fastf1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data.fastf1_loader import load_session, list_available_races


def main():
    config = get_config()
    races = list_available_races()

    print("=" * 60)
    print("  TrackShift — FastF1 Data Download")
    print("=" * 60)

    for race in races:
        print(f"\n→ Downloading {race['label']}...")
        laps, weather, error = load_session(
            race["year"], race["gp"], "R",
            cache_dir=str(config.raw_dir),
        )
        if error:
            print(f"  ✗ Failed: {error}")
        else:
            n_laps = len(laps) if laps is not None else 0
            has_weather = weather is not None and len(weather) > 0
            print(f"  ✓ {n_laps} laps, weather={'yes' if has_weather else 'no'}")

    print("\n✓ Download complete. Data cached in data/raw/")


if __name__ == "__main__":
    main()
