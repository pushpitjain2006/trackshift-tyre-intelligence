#!/usr/bin/env bash
# TrackShift Tyre Degradation Intelligence — Launch Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  TRACKSHIFT TYRE DEGRADATION INTELLIGENCE"
echo "=========================================="
echo ""

# Ensure data directories exist
mkdir -p data/{raw,processed,demo} models/cached

# Launch Streamlit
exec streamlit run app/dashboard.py \
    --server.headless true \
    --theme.base dark \
    --theme.primaryColor "#FF1744" \
    --browser.gatherUsageStats false \
    "$@"
