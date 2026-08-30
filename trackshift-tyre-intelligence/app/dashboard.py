"""
TrackShift Tyre Degradation Intelligence — Main Dashboard.

Streamlit application orchestrating the complete demo experience:
from data loading through model fitting to insight generation.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import logging
import numpy as np
import pandas as pd
import streamlit as st

from app.styling import CUSTOM_CSS, COLORS
from app.components import (
    chart_confounders,
    chart_degradation,
    chart_model_comparison,
    chart_raw_laptimes,
    render_data_quality,
    render_diagnostics,
    render_engineering_insight,
    render_ev_section,
    render_how_it_works,
    render_kpi_row,
    render_model_comparison_table,
)
from src.config import Config, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TrackShift — Tyre Degradation Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading helpers (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_fastf1_data(year: int, gp: str, session_type: str):
    """Load and preprocess FastF1 session data."""
    from src.data.fastf1_loader import load_session
    laps_df, weather_df, error = load_session(year, gp, session_type,
                                               cache_dir=str(_project_root / "data" / "raw"))
    if error:
        return None, None, None, error

    from src.data.preprocessing import clean_laps
    config = get_config()
    clean_df, summary = clean_laps(laps_df, config)
    return clean_df, weather_df, summary, None


@st.cache_data(show_spinner=False)
def load_demo_data():
    """Load synthetic demo data."""
    from src.data.demo_data import load_demo_dataset, generate_demo_metadata
    df = load_demo_dataset()
    meta = generate_demo_metadata()
    from src.data.preprocessing import PreprocessingSummary
    summary = PreprocessingSummary(raw_count=len(df), valid_count=len(df))
    return df, meta, summary


@st.cache_data(show_spinner=False)
def get_stints_for_driver(clean_df_hash, clean_df_json, driver):
    """Extract stints for a driver (cached by hash)."""
    clean_df = pd.read_json(clean_df_json)
    from src.features.tyre_features import extract_stints
    return extract_stints(clean_df, driver)


@st.cache_data(show_spinner=False)
def build_features(stint_df_json, weather_json, fuel_kg, burn_rate):
    """Build feature matrix (cached)."""
    stint_df = pd.read_json(stint_df_json)
    weather_df = pd.read_json(weather_json) if weather_json else None
    config = get_config()
    config.fuel.initial_fuel_kg = fuel_kg
    config.fuel.fuel_burn_per_lap_kg = burn_rate
    from src.features.confounders import build_feature_matrix
    return build_feature_matrix(stint_df, weather_df, config)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    """Render sidebar controls and return user selections."""
    with st.sidebar:
        st.markdown("## 🏎️ TRACKSHIFT")
        st.markdown("##### Tyre Degradation Intelligence")
        st.divider()

        # Data source
        data_source = st.radio(
            "Data Source",
            ["🎯 Demo Mode", "📡 FastF1 (Live)"],
            index=0,
            help="Demo Mode uses cached data for reliable demo. FastF1 downloads real F1 data.",
        )

        is_demo = "Demo" in data_source

        # Race selection (FastF1 mode)
        race_config = {}
        if not is_demo:
            from src.data.fastf1_loader import list_available_races
            races = list_available_races()
            race_labels = [r["label"] for r in races]
            selected_idx = st.selectbox("Race", range(len(race_labels)),
                                         format_func=lambda i: race_labels[i])
            race_config = races[selected_idx]

        st.divider()

        # Model choice
        model_choice = st.selectbox(
            "Model",
            ["Bayesian State-Space", "Fast State-Space (Kalman)", "Multivariate Regression"],
            index=0,
        )
        model_map = {
            "Bayesian State-Space": "bayesian",
            "Fast State-Space (Kalman)": "kalman",
            "Multivariate Regression": "regression",
        }

        st.divider()

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            fuel_kg = st.slider("Initial Fuel (kg)", 90.0, 115.0, 110.0, 1.0,
                                help="Estimated initial fuel load. Exact values not publicly observable.")
            burn_rate = st.slider("Fuel Burn Rate (kg/lap)", 1.2, 2.5, 1.8, 0.1,
                                  help="Estimated fuel consumption per lap.")
            draws = st.slider("MCMC Draws", 200, 2000, 500, 100)
            tune = st.slider("MCMC Tune", 200, 2000, 500, 100)

        return {
            "is_demo": is_demo,
            "race_config": race_config,
            "model_choice": model_map[model_choice],
            "model_label": model_choice,
            "fuel_kg": fuel_kg if not is_demo else 110.0,
            "burn_rate": burn_rate if not is_demo else 1.8,
            "draws": draws if not is_demo else 500,
            "tune": tune if not is_demo else 500,
        }


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    selections = render_sidebar()

    # ── HEADER ──
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="font-size: 2.4em; font-weight: 800; margin-bottom: 0;
                    background: linear-gradient(135deg, #FF1744, #FF6B6B);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            TRACKSHIFT
        </h1>
        <h3 style="color: #B0B8C8; font-weight: 400; margin-top: 4px;">
            Tyre Degradation Intelligence
        </h3>
        <p style="color: #708090; font-size: 0.95em; font-style: italic;">
            Separating tyre wear from the noise hiding it.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── LOAD DATA ──
    if selections["is_demo"]:
        demo_df, demo_meta, preprocess_summary = load_demo_data()
        if demo_df is None:
            st.error("Failed to load demo data.")
            return

        st.markdown(
            f'<span class="status-badge status-demo">DEMO DATA</span> '
            f'{demo_meta["data_source"]}',
            unsafe_allow_html=True,
        )

        # For demo, data is already featured
        clean_df = demo_df
        weather_df = None
        data_source_label = demo_meta["data_source"]
        available_drivers = clean_df["Driver"].unique().tolist()

    else:
        rc = selections["race_config"]
        with st.spinner(f"Loading {rc.get('label', 'race')} via FastF1..."):
            clean_df, weather_df, preprocess_summary, error = load_fastf1_data(
                rc["year"], rc["gp"], "R"
            )
        if error:
            st.warning(f"⚠️ FastF1 unavailable: {error}")
            st.info("Switching to Demo Mode...")
            demo_df, demo_meta, preprocess_summary = load_demo_data()
            clean_df = demo_df
            weather_df = None
            data_source_label = f"FALLBACK: {demo_meta['data_source']}"
            available_drivers = clean_df["Driver"].unique().tolist()
            selections["is_demo"] = True
        else:
            st.markdown(
                '<span class="status-badge status-live">REAL FASTF1 DATA</span>',
                unsafe_allow_html=True,
            )
            data_source_label = "REAL FASTF1 DATA"
            from src.data.fastf1_loader import list_drivers
            available_drivers = list_drivers(clean_df)

    # ── DRIVER & STINT SELECTION ──
    col_drv, col_stint = st.columns(2)

    if selections["is_demo"]:
        # Demo mode: auto-select, but allow override
        driver = col_drv.selectbox("Driver", available_drivers, index=0)

        if "tyre_age" in clean_df.columns:
            # Demo data is pre-featured — treat as single stint
            stint_options = ["Stint 1 (Demo)"]
            stint_idx = col_stint.selectbox("Stint", range(len(stint_options)),
                                            format_func=lambda i: stint_options[i])
            stint_df = clean_df.copy()
            compound = clean_df["Compound"].iloc[0] if "Compound" in clean_df.columns else "MEDIUM"
            stint_label = "Demo Stint"
        else:
            stint_df, compound, stint_label = _select_stint(clean_df, driver, col_stint)
            if stint_df is None:
                return
    else:
        driver = col_drv.selectbox("Driver", available_drivers)
        stint_df, compound, stint_label = _select_stint(clean_df, driver, col_stint, weather_df,
                                                         selections["fuel_kg"], selections["burn_rate"])
        if stint_df is None:
            return

    # ── DATA QUALITY ──
    render_data_quality(preprocess_summary)

    st.divider()

    # ── SECTION 1: THE PARADOX ──
    st.markdown("""
    <div class="section-header">
        <h2>🤔 Why is the car getting faster if the tyres are degrading?</h2>
        <p>Raw lap time is confounded by multiple factors changing simultaneously</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick baselines (always fast)
    from src.models.baseline import fit_naive, fit_multivariate
    naive_result = fit_naive(stint_df)

    fig_raw = chart_raw_laptimes(stint_df, naive_result.predictions, compound)
    st.plotly_chart(fig_raw, use_container_width=True)

    # Annotation
    naive_slope = naive_result.degradation_rate_per_lap
    if naive_slope < 0:
        st.markdown(f"""
        > **📉 The naive model says: the car is getting faster** ({naive_slope:+.3f} s/lap).
        > But that's because **fuel is burning off** — the car loses ~1.8 kg/lap,
        > making it lighter and faster, masking tyre degradation.
        """)
    else:
        st.markdown(f"""
        > The naive model estimates degradation at {naive_slope:+.3f} s/lap — but this
        > estimate is **confounded** by fuel mass, temperature, and track evolution.
        """)

    # ── SECTION 2: CONFOUNDERS ──
    st.markdown("""
    <div class="section-header">
        <h2>📊 The Confounders</h2>
        <p>These variables change lap time independently of tyre wear</p>
    </div>
    """, unsafe_allow_html=True)

    fig_fuel, fig_temp, fig_prog = chart_confounders(stint_df)
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(fig_fuel, use_container_width=True)
    c2.plotly_chart(fig_temp, use_container_width=True)
    c3.plotly_chart(fig_prog, use_container_width=True)

    st.markdown("""
    > ⛽ **Fuel** ↓ → Car lighter → Lap time ↓ &nbsp;&nbsp;|&nbsp;&nbsp;
    > 🌡️ **Temperature** changes grip &nbsp;&nbsp;|&nbsp;&nbsp;
    > 🛣️ **Track evolution** improves surface grip
    >
    > Meanwhile: **Tyre grip** ↓ (hidden signal)
    """)

    st.divider()

    # ── SECTION 3: DEGRADATION ENGINE ──
    st.markdown("""
    <div class="section-header">
        <h2>🔬 The Degradation-State Engine</h2>
        <p>Isolating the hidden tyre degradation signal</p>
    </div>
    """, unsafe_allow_html=True)

    # Run model
    run_key = f"model_run_{driver}_{stint_label}_{selections['model_choice']}"

    if st.button("🚀 ISOLATE TYRE DEGRADATION", type="primary", use_container_width=True):
        st.session_state[run_key] = True

    if st.session_state.get(run_key, False):
        with st.spinner("Running degradation analysis..."):
            config = get_config()
            config.fuel.initial_fuel_kg = selections["fuel_kg"]
            config.fuel.fuel_burn_per_lap_kg = selections["burn_rate"]
            config.sampling.draws = selections["draws"]
            config.sampling.tune = selections["tune"]

            multi_result = fit_multivariate(stint_df)

            from src.models.inference import run_degradation_analysis
            from src.data.cache import cache_key, config_hash

            ck = cache_key(
                data_source_label, driver, 0, config_hash(config)
            )

            bayesian_result, _, _, model_used = run_degradation_analysis(
                stint_df, config, selections["model_choice"], cache_key_str=ck
            )

        # KPI Row
        if "Kalman" in model_used or "Regression" in model_used:
            confidence = "N/A (MLE)"
        else:
            confidence = 85.0
            if bayesian_result.diagnostics:
                rhat = bayesian_result.diagnostics.get("max_rhat", 1.0)
                if isinstance(rhat, (int, float)):
                    confidence = max(0, min(100, 100 - (rhat - 1.0) * 2000))

        render_kpi_row(
            deg_rate=bayesian_result.rate_per_lap,
            total_deg=bayesian_result.total_degradation,
            stint_laps=len(stint_df),
            confidence=confidence,
            compound=compound,
        )

        st.markdown("")

        # Hero chart
        fig_deg = chart_degradation(
            bayesian_result.tyre_age,
            bayesian_result.degradation_mean,
            bayesian_result.degradation_p05,
            bayesian_result.degradation_p95,
            compound,
        )
        st.plotly_chart(fig_deg, use_container_width=True)

        st.caption(f"**Inference mode:** {model_used} | "
                   f"**Sampling time:** {bayesian_result.sampling_time_sec:.1f}s")

        st.divider()

        # ── SECTION 4: MODEL COMPARISON ──
        st.markdown("""
        <div class="section-header">
            <h2>⚖️ Model Comparison</h2>
            <p>Progression from naive to confounder-aware to latent state estimation</p>
        </div>
        """, unsafe_allow_html=True)

        # Build degradation curves for comparison
        tyre_age = stint_df["tyre_age"].values
        naive_deg = naive_result.degradation_rate_per_lap * (tyre_age - tyre_age[0])
        multi_deg = multi_result.degradation_rate_per_lap * (tyre_age - tyre_age[0])

        fig_comp = chart_model_comparison(
            tyre_age, naive_deg, multi_deg, bayesian_result.degradation_mean
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        render_model_comparison_table(naive_result, multi_result, bayesian_result)

        st.divider()

        # ── SECTION 5: ENGINEERING INSIGHT ──
        render_engineering_insight(bayesian_result, naive_result, multi_result)

        st.divider()

        # ── DIAGNOSTICS ──
        render_diagnostics(bayesian_result.diagnostics, bayesian_result.sampling_time_sec)

    # ── HOW IT WORKS ──
    render_how_it_works()

    st.divider()

    # ── EV SECTION ──
    render_ev_section()

    # ── FOOTER ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #708090; font-size: 0.8em;'>"
        "TrackShift Tyre Degradation Intelligence — Innovation Challenge 2026<br>"
        "Built with FastF1, PyMC, Plotly, and Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


def _select_stint(clean_df, driver, col_stint, weather_df=None, fuel_kg=110.0, burn_rate=1.8):
    """Helper to select and feature-engineer a stint."""
    from src.features.tyre_features import extract_stints, get_stint_df
    from src.features.confounders import build_feature_matrix

    stints = extract_stints(clean_df, driver)
    if not stints:
        st.warning(f"No usable stints found for {driver}.")
        return None, "UNKNOWN", ""

    stint_labels = [s.label for s in stints]
    selected_idx = col_stint.selectbox("Stint", range(len(stint_labels)),
                                        format_func=lambda i: stint_labels[i])
    selected_stint = stints[selected_idx]

    # Get stint data
    stint_df = get_stint_df(clean_df, selected_stint)

    # Feature engineering
    config = get_config()
    config.fuel.initial_fuel_kg = fuel_kg
    config.fuel.fuel_burn_per_lap_kg = burn_rate
    stint_df = build_feature_matrix(stint_df, weather_df, config)

    return stint_df, selected_stint.compound, selected_stint.label


if __name__ == "__main__":
    main()
