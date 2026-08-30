"""
F1 → EV Translation module.

Maps the tyre degradation intelligence framework to electric vehicle
battery State of Health (SoH) estimation. Same mathematical abstraction:
infer hidden system health from noisy, confounded observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_ev_analogy_data() -> dict:
    """Return the F1 → EV mapping for display in the dashboard."""
    return {
        "mappings": [
            {
                "f1_concept": "Tyre Degradation",
                "f1_detail": "Hidden grip loss over stint",
                "ev_concept": "Battery Degradation / SoH",
                "ev_detail": "Hidden capacity loss over vehicle life",
                "icon_f1": "🏎️",
                "icon_ev": "🔋",
            },
            {
                "f1_concept": "Fuel Mass",
                "f1_detail": "110 kg → 0 kg (car gets lighter)",
                "ev_concept": "Payload / Cargo",
                "ev_detail": "Variable passenger/cargo weight",
                "icon_f1": "⛽",
                "icon_ev": "📦",
            },
            {
                "f1_concept": "Track Temperature",
                "f1_detail": "Surface temp affects tyre grip",
                "ev_concept": "Ambient Temperature",
                "ev_detail": "Heat affects battery chemistry & AC drain",
                "icon_f1": "🌡️",
                "icon_ev": "☀️",
            },
            {
                "f1_concept": "Lap Performance",
                "f1_detail": "Lap time (seconds)",
                "ev_concept": "Energy / Range Performance",
                "ev_detail": "kWh/km or range per charge",
                "icon_f1": "⏱️",
                "icon_ev": "📊",
            },
            {
                "f1_concept": "Track Evolution",
                "f1_detail": "Circuit grip improves over session",
                "ev_concept": "Route / Traffic Conditions",
                "ev_detail": "Urban vs highway, stop-and-go",
                "icon_f1": "🛣️",
                "icon_ev": "🚦",
            },
        ],
        "abstraction": (
            "Hidden Health State Estimation Under Observational Confounding"
        ),
        "impact": {
            "sector": "Indian commercial EV fleets (BluSmart, Delhivery, Amazon India)",
            "problem": (
                "Battery SoH estimates are corrupted by payload variance and extreme "
                "temperatures. Operators cannot distinguish real degradation from noise."
            ),
            "solution": (
                "Apply the same Bayesian state-space model to isolate true battery "
                "degradation from operational confounders — enabling accurate "
                "predictive maintenance."
            ),
            "impact_value": (
                "10–15% extension of effective battery lifespan by eliminating "
                "premature replacements. For a 5,000-vehicle fleet at ₹3,00,000 "
                "per battery pack: ~₹15 Crore annual CapEx deferral."
            ),
        },
    }


def generate_mock_ev_data(seed: int = 123, n_trips: int = 50) -> pd.DataFrame:
    """Generate mock EV fleet telemetry for demonstration.

    Simulates the same hidden-state problem: apparent energy consumption
    is confounded by payload and temperature, masking true battery degradation.

    Returns
    -------
    pd.DataFrame
        Mock EV telemetry with known ground truth battery health.
    """
    rng = np.random.RandomState(seed)

    trip_ids = np.arange(1, n_trips + 1)
    distance_km = rng.uniform(25, 80, n_trips)
    payload_kg = rng.uniform(100, 500, n_trips)
    ambient_temp = rng.uniform(25, 48, n_trips)  # Indian summer

    # True battery degradation (gradual, hidden)
    true_battery_health = 1.0 - 0.003 * trip_ids + rng.normal(0, 0.002, n_trips)
    true_battery_health = np.clip(true_battery_health, 0.7, 1.0)

    # Observed energy consumption (confounded)
    base_consumption = 0.15  # kWh/km
    energy_per_km = (
        base_consumption
        + 0.00005 * payload_kg          # heavier = more energy
        + 0.001 * (ambient_temp - 30)    # AC drain
        + 0.05 * (1 - true_battery_health)  # degradation effect
        + rng.normal(0, 0.005, n_trips)
    )
    energy_used = energy_per_km * distance_km
    voltage = 400 - 15 * (1 - true_battery_health) + rng.normal(0, 2, n_trips)
    current = energy_used * 1000 / voltage  # rough approximation

    return pd.DataFrame({
        "vehicle_id": "EV-001",
        "trip_id": trip_ids,
        "distance_km": distance_km.round(1),
        "payload_kg": payload_kg.round(0),
        "ambient_temp_C": ambient_temp.round(1),
        "energy_used_kWh": energy_used.round(2),
        "energy_per_km_kWh": energy_per_km.round(4),
        "voltage": voltage.round(1),
        "current": current.round(1),
        "estimated_battery_health": true_battery_health.round(4),
    })
