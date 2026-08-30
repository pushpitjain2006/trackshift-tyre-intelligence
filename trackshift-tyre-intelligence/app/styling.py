"""
Dashboard styling — dark motorsport aesthetic.

Color palette, CSS, and Plotly template for a polished engineering tool feel.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0E1117",
    "bg_secondary": "#1A1F2E",
    "bg_card": "#1E2333",
    "text": "#FAFAFA",
    "text_secondary": "#B0B8C8",
    "accent": "#FF1744",
    "accent_green": "#00E676",
    "accent_blue": "#448AFF",
    "accent_gold": "#FFD740",
    "success": "#00C853",
    "warning": "#FF9100",
    "error": "#FF1744",
    "grid": "#2A3040",
}

COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC107",
    "HARD": "#EEEEEE",
    "INTERMEDIATE": "#43A047",
    "WET": "#1E88E5",
    "UNKNOWN": "#AAAAAA",
}

# ---------------------------------------------------------------------------
# Plotly dark template
# ---------------------------------------------------------------------------
def get_plotly_template() -> dict:
    """Return a Plotly layout template for the motorsport dark theme."""
    return dict(
        paper_bgcolor=COLORS["bg_secondary"],
        plot_bgcolor=COLORS["bg_secondary"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif", size=13),
        xaxis=dict(
            gridcolor=COLORS["grid"],
            gridwidth=0.5,
            zerolinecolor=COLORS["grid"],
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            gridwidth=0.5,
            zerolinecolor=COLORS["grid"],
            tickfont=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor=COLORS["bg_card"],
            font_color=COLORS["text"],
            font_size=12,
        ),
    )


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global dark theme */
    .stApp {
        background-color: #0E1117;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1E2333 0%, #252B3B 100%);
        border: 1px solid #2A3040;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.3s;
    }
    .kpi-card:hover {
        border-color: #FF1744;
    }
    .kpi-value {
        font-size: 2.0em;
        font-weight: 700;
        color: #FF1744;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.85em;
        color: #B0B8C8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-unit {
        font-size: 0.75em;
        color: #708090;
    }

    /* Section headers */
    .section-header {
        border-left: 4px solid #FF1744;
        padding-left: 16px;
        margin: 32px 0 16px 0;
    }
    .section-header h2 {
        color: #FAFAFA;
        font-size: 1.4em;
        margin-bottom: 4px;
    }
    .section-header p {
        color: #B0B8C8;
        font-size: 0.9em;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .status-live { background: #00C853; color: #0E1117; }
    .status-demo { background: #FFD740; color: #0E1117; }
    .status-synthetic { background: #448AFF; color: white; }

    /* EV translation table */
    .ev-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
    }
    .ev-table td {
        padding: 12px 16px;
        background: #1E2333;
        border: none;
    }
    .ev-table td:first-child { border-radius: 8px 0 0 8px; }
    .ev-table td:last-child { border-radius: 0 8px 8px 0; }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
