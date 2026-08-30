"""
Reusable dashboard UI components.

All chart builders and UI elements used by dashboard.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.styling import COLORS, COMPOUND_COLORS, get_plotly_template


# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
def render_kpi_row(
    deg_rate: float,
    total_deg: float,
    stint_laps: int,
    confidence: float | str,
    compound: str = "MEDIUM",
) -> None:
    """Render the top KPI summary row."""
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Tyre Degradation Rate</div>
            <div class="kpi-value">{deg_rate:+.3f}</div>
            <div class="kpi-unit">seconds / lap</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Stint Degradation</div>
            <div class="kpi-value">{total_deg:+.2f}</div>
            <div class="kpi-unit">seconds</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Stint Length</div>
            <div class="kpi-value">{stint_laps}</div>
            <div class="kpi-unit">laps</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        if isinstance(confidence, (int, float)):
            color = COLORS["success"] if confidence > 70 else COLORS["warning"]
            val_str = f"{confidence:.0f}%"
            size = "2.0em"
        else:
            color = COLORS["text_secondary"]
            val_str = str(confidence)
            size = "1.5em"
            
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Confidence</div>
            <div class="kpi-value" style="color: {color}; font-size: {size};">{val_str}</div>
            <div class="kpi-unit">posterior quality</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_raw_laptimes(
    stint_df: pd.DataFrame,
    naive_predictions: np.ndarray | None = None,
    compound: str = "MEDIUM",
) -> go.Figure:
    """Raw lap time vs tyre age — the paradox chart."""
    layout = get_plotly_template()
    compound_color = COMPOUND_COLORS.get(compound, "#AAAAAA")

    fig = go.Figure()

    # Raw observations
    fig.add_trace(go.Scatter(
        x=stint_df["tyre_age"],
        y=stint_df["LapTime_sec"],
        mode="markers+lines",
        name="Raw Lap Time",
        marker=dict(color=compound_color, size=9, line=dict(width=1, color="white")),
        line=dict(color=compound_color, width=1.5, dash="dot"),
        hovertemplate="Tyre Age: %{x}<br>Lap Time: %{y:.3f}s<extra></extra>",
    ))

    # Naive regression line
    if naive_predictions is not None:
        fig.add_trace(go.Scatter(
            x=stint_df["tyre_age"],
            y=naive_predictions,
            mode="lines",
            name="Naive Trend (misleading)",
            line=dict(color="#FF6B6B", width=2.5, dash="dash"),
            hovertemplate="Naive prediction: %{y:.3f}s<extra></extra>",
        ))

    fig.update_layout(
        **layout,
        title=dict(
            text="Raw Lap Time vs Tyre Age — The Paradox",
            font=dict(size=16, color=COLORS["text"]),
        ),
        xaxis_title="Tyre Age (laps)",
        yaxis_title="Lap Time (seconds)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
        height=420,
    )
    return fig


def chart_confounders(stint_df: pd.DataFrame) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Three compact confounder charts: fuel, temperature, track progress."""
    layout = get_plotly_template()
    layout["height"] = 280
    layout["margin"] = dict(l=50, r=20, t=40, b=40)

    # Fuel mass
    fig_fuel = go.Figure()
    fig_fuel.add_trace(go.Scatter(
        x=stint_df["tyre_age"],
        y=stint_df["fuel_mass_kg"],
        mode="lines+markers",
        name="Fuel Mass",
        line=dict(color=COLORS["accent_gold"], width=2.5),
        marker=dict(size=5, color=COLORS["accent_gold"]),
        fill="tozeroy",
        fillcolor="rgba(255, 215, 64, 0.1)",
    ))
    fig_fuel.update_layout(**layout, title="Estimated Fuel Mass",
                           xaxis_title="Tyre Age", yaxis_title="kg")

    # Track temperature
    fig_temp = go.Figure()
    if "track_temp_C" in stint_df.columns:
        fig_temp.add_trace(go.Scatter(
            x=stint_df["tyre_age"],
            y=stint_df["track_temp_C"],
            mode="lines+markers",
            name="Track Temp",
            line=dict(color=COLORS["accent"], width=2.5),
            marker=dict(size=5, color=COLORS["accent"]),
        ))
    fig_temp.update_layout(**layout, title="Track Temperature",
                           xaxis_title="Tyre Age", yaxis_title="°C")

    # Track progress
    fig_prog = go.Figure()
    if "track_progress" in stint_df.columns:
        fig_prog.add_trace(go.Scatter(
            x=stint_df["tyre_age"],
            y=stint_df["track_progress"],
            mode="lines+markers",
            name="Track Evolution",
            line=dict(color=COLORS["accent_green"], width=2.5),
            marker=dict(size=5, color=COLORS["accent_green"]),
            fill="tozeroy",
            fillcolor="rgba(0, 230, 118, 0.1)",
        ))
    fig_prog.update_layout(**layout, title="Session Progression",
                           xaxis_title="Tyre Age", yaxis_title="Normalized (0–1)")

    return fig_fuel, fig_temp, fig_prog


def chart_degradation(
    tyre_age: np.ndarray,
    deg_mean: np.ndarray,
    deg_p05: np.ndarray,
    deg_p95: np.ndarray,
    compound: str = "MEDIUM",
) -> go.Figure:
    """Hero visualization: estimated latent tyre degradation with uncertainty."""
    layout = get_plotly_template()
    compound_color = COMPOUND_COLORS.get(compound, "#AAAAAA")

    fig = go.Figure()

    # Credible interval band
    fig.add_trace(go.Scatter(
        x=np.concatenate([tyre_age, tyre_age[::-1]]),
        y=np.concatenate([deg_p95, deg_p05[::-1]]),
        fill="toself",
        fillcolor="rgba(255, 23, 68, 0.15)",
        line=dict(color="rgba(255,23,68,0)"),
        name="90% Credible Interval",
        hoverinfo="skip",
    ))

    # Posterior mean
    fig.add_trace(go.Scatter(
        x=tyre_age,
        y=deg_mean,
        mode="lines+markers",
        name="Estimated Degradation",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=7, color=COLORS["accent"], line=dict(width=1, color="white")),
        hovertemplate="Tyre Age: %{x}<br>Degradation: %{y:+.3f}s<extra></extra>",
    ))

    # Zero reference line
    fig.add_hline(
        y=0, line_dash="dot", line_color=COLORS["text_secondary"],
        annotation_text="Lap 1 baseline",
        annotation_position="bottom right",
        annotation_font_color=COLORS["text_secondary"],
    )

    fig.update_layout(
        **layout,
        title=dict(
            text="🏎️ Estimated Latent Tyre Degradation",
            font=dict(size=18, color=COLORS["text"]),
        ),
        xaxis_title="Tyre Age (laps)",
        yaxis_title="Performance Degradation (seconds, relative to Lap 1)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
        height=480,
    )
    return fig


def chart_model_comparison(
    tyre_age: np.ndarray,
    naive_deg: np.ndarray,
    multi_deg: np.ndarray,
    bayesian_deg: np.ndarray,
) -> go.Figure:
    """Compare degradation estimates from all three models."""
    layout = get_plotly_template()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tyre_age, y=naive_deg,
        mode="lines", name="Naive (TyreAge only)",
        line=dict(color="#FF6B6B", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=tyre_age, y=multi_deg,
        mode="lines", name="Multivariate (confounder-aware)",
        line=dict(color=COLORS["accent_blue"], width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=tyre_age, y=bayesian_deg,
        mode="lines+markers", name="Bayesian State-Space",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=6),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_secondary"])

    fig.update_layout(
        **layout,
        title="Model Comparison — Degradation Estimates",
        xaxis_title="Tyre Age (laps)",
        yaxis_title="Estimated Degradation (seconds)",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Text sections
# ---------------------------------------------------------------------------
def render_model_comparison_table(naive_result, multi_result, bayesian_result) -> None:
    """Render the comparison table for all three models."""
    data = {
        "Model": [
            naive_result.model_name if hasattr(naive_result, 'model_name') else "Naive",
            multi_result.model_name if hasattr(multi_result, 'model_name') else "Multivariate",
            bayesian_result.model_name if hasattr(bayesian_result, 'model_name') else "Bayesian SS",
        ],
        "Fuel Accounted?": ["❌ No", "✅ Yes", "✅ Yes"],
        "Temperature?": ["❌ No", "✅ Yes", "✅ Yes"],
        "Latent State?": ["❌ No", "❌ No", "✅ Yes"],
        "Uncertainty?": ["❌ No", "❌ No", "✅ Yes"],
        "Deg. Rate (s/lap)": [
            f"{getattr(naive_result, 'degradation_rate_per_lap', getattr(naive_result, 'rate_per_lap', 0)):+.4f}",
            f"{getattr(multi_result, 'degradation_rate_per_lap', getattr(multi_result, 'rate_per_lap', 0)):+.4f}",
            f"{bayesian_result.rate_per_lap:+.4f}",
        ],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def render_engineering_insight(bayesian_result, naive_result, multi_result) -> None:
    """Generate a natural-language engineering summary using actual model numbers."""
    deg_rate = bayesian_result.rate_per_lap
    total_deg = bayesian_result.total_degradation
    naive_rate = getattr(naive_result, 'degradation_rate_per_lap', getattr(naive_result, 'rate_per_lap', 0))

    fuel_coef = bayesian_result.coefficients.get("beta_fuel_per_kg", 0)
    mode = bayesian_result.inference_mode

    direction = "upward" if deg_rate > 0 else "negligible"

    st.markdown(f"""
    ### 📋 Engineering Summary

    **Tyre degradation** is estimated at approximately **{deg_rate:+.4f} seconds per lap**
    ({total_deg:+.3f} s total across the stint).

    The **raw lap-time trend is misleading** because the naive model estimates a degradation
    rate of {naive_rate:+.4f} s/lap — {'suggesting the car is actually getting *faster*' if naive_rate < 0 else 'underestimating the true rate'}.
    This is primarily because fuel mass decreases throughout the stint, making the car
    lighter and masking the underlying tyre wear.

    After accounting for estimated fuel mass{' (fuel coefficient: ' + f'{fuel_coef:+.4f} s/kg' + ')' if fuel_coef != 0 else ''}
    and environmental effects, the latent tyre state shows a **{direction} degradation trend**.

    **Inference mode:** {mode}

    > ⚠️ **Scientific note:** Fuel mass is estimated, not measured. Track temperature is
    > observational. The latent state represents *performance degradation attributable to
    > tyre wear*, not direct measurement of rubber loss.
    """)


def render_data_quality(summary) -> None:
    """Render preprocessing data quality in an expandable section."""
    with st.expander("📊 Data Quality Report", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Raw Laps", summary.raw_count)
        col2.metric("Valid Laps", summary.valid_count)
        col3.metric("Removed", summary.removed_count)

        if summary.removal_reasons:
            st.markdown("**Removal Breakdown:**")
            for reason, count in sorted(summary.removal_reasons.items(), key=lambda x: -x[1]):
                st.markdown(f"- {reason}: **{count}**")


def render_diagnostics(diagnostics: dict, sampling_time: float = 0) -> None:
    """Render model diagnostics in an expandable section."""
    with st.expander("🔬 Model Diagnostics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max R-hat", f"{diagnostics.get('max_rhat', 'N/A'):.3f}"
                     if isinstance(diagnostics.get('max_rhat'), (int, float)) else "N/A")
        col2.metric("Min ESS", f"{diagnostics.get('min_ess', 'N/A'):.0f}"
                     if isinstance(diagnostics.get('min_ess'), (int, float)) else "N/A")
        col3.metric("Divergences", diagnostics.get('divergences', 'N/A'))
        col4.metric("Sampling Time", f"{sampling_time:.1f}s")

        rhat = diagnostics.get('max_rhat', 1.0)
        if isinstance(rhat, (int, float)):
            if rhat < 1.01:
                st.success("✅ Sampling converged (R-hat < 1.01)")
            elif rhat < 1.05:
                st.warning("⚠️ Marginal convergence (R-hat < 1.05)")
            else:
                st.error("❌ Poor convergence (R-hat ≥ 1.05)")

        method = diagnostics.get("method", "")
        if method:
            st.info(f"Inference method: {method}")


def render_ev_section() -> None:
    """Render the F1 → EV / India translation section."""
    from src.ev.translation import get_ev_analogy_data

    data = get_ev_analogy_data()

    st.markdown("""
    <div class="section-header">
        <h2>🇮🇳 Motorsport → India: Battery Health Intelligence</h2>
        <p>The same mathematical abstraction applied to electric vehicles</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Transferable Principle:** *{data['abstraction']}*")

    # Mapping table
    cols = st.columns([1, 1, 1, 1])
    cols[0].markdown("**F1 Concept**")
    cols[1].markdown("**F1 Detail**")
    cols[2].markdown("**EV Concept**")
    cols[3].markdown("**EV Detail**")

    for m in data["mappings"]:
        cols = st.columns([1, 1, 1, 1])
        cols[0].markdown(f"{m['icon_f1']} {m['f1_concept']}")
        cols[1].markdown(f"_{m['f1_detail']}_")
        cols[2].markdown(f"{m['icon_ev']} {m['ev_concept']}")
        cols[3].markdown(f"_{m['ev_detail']}_")

    st.divider()

    # Impact
    impact = data["impact"]
    st.markdown(f"""
    **Target Sector:** {impact['sector']}

    **Problem:** {impact['problem']}

    **Solution:** {impact['solution']}

    **Quantified Impact:** {impact['impact_value']}
    """)


def render_how_it_works() -> None:
    """Render the model explanation section."""
    with st.expander("🧠 How It Works — Model Architecture", expanded=False):
        st.markdown("""
        **Observed Lap Time** is decomposed as:

        ```
        Lap Time = Base Performance
                 + Fuel Effect          ← car gets lighter → faster
                 + Temperature Effect   ← surface grip changes
                 + Track Evolution      ← circuit rubbers in
                 + Tyre Degradation     ← HIDDEN, must be inferred
                 + Noise                ← driver errors, traffic
        ```

        **The tyre component is not directly observable.** We infer it as a
        *latent random walk with drift* using a Bayesian state-space model:

        **Observation equation:**
        `y_t = β₀ + β_f·fuel_t + β_c·temp_t + β_p·progress_t + α_t + ε_t`

        **State transition:**
        `α_t = α_{t-1} + δ + ω_t`

        Where `δ ≥ 0` is the degradation drift, constrained to be non-negative
        (tyres generally degrade, not improve).

        **Inference:** Posterior samples via NUTS (No-U-Turn Sampler) with
        Student-T observation noise for robustness against outliers (lockups, traffic).

        **Uncertainty:** The shaded region represents the 90% posterior credible
        interval — quantifying how confident we are in the hidden degradation state.
        """)
