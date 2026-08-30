# 🏎️ TrackShift — Tyre Degradation Intelligence

> **TrackShift Innovation Challenge 2026**
>
> *Separating tyre wear from the noise hiding it.*

## The Problem

In Formula 1, raw lap times during a stint can appear **flat or even decreasing** — suggesting the car is getting *faster* as tyre age increases. This is paradoxical: tyres physically degrade and lose grip over time.

The explanation is **confounding variables**. Multiple effects simultaneously influence lap time:

| Factor | Direction | Effect |
|--------|-----------|--------|
| Tyre degradation | ↑ lap time | Grip loss → slower |
| Fuel burn | ↓ lap time | Lighter car → faster |
| Track evolution | ↓ lap time | Rubber on surface → more grip |
| Temperature variation | ↕ lap time | Affects tyre/aero performance |
| Driver noise | ↕ lap time | Lockups, traffic, mistakes |

Simply regressing `LapTime ~ TyreAge` produces **misleading results** because it conflates these independent effects.

## Our Solution

We treat **tyre degradation as a hidden (latent) state** that cannot be directly observed, but can be **inferred** from the sequence of lap times after accounting for observable confounders.

### Mathematical Model

**Observation equation:**

```
y_t = β₀ + β_f·f_t + β_c·c_t + β_p·p_t + α_t + ε_t
```

**State transition:**

```
α_t = α_{t-1} + δ + ω_t
```

Where:

| Symbol | Meaning |
|--------|---------|
| `y_t` | Observed lap time at lap *t* (seconds) |
| `f_t` | Estimated fuel mass (kg) — decreases linearly |
| `c_t` | Track temperature (°C) — from weather data |
| `p_t` | Session progression (0–1) — track evolution proxy |
| `α_t` | **Latent tyre degradation state** (inferred) |
| `δ` | Degradation drift rate (constrained ≥ 0) |
| `ε_t` | Observation noise (Student-T for robustness) |
| `ω_t` | Process noise |

### Why `LapTime ~ TyreAge` Is Insufficient

A naive regression captures the **net** effect of all factors. When fuel burn (~1.8 kg/lap × ~0.035 s/kg = ~0.063 s/lap time reduction) exceeds tyre degradation (~0.05 s/lap increase), the naive slope can be **negative** — incorrectly suggesting tyres are *improving*.

Our Bayesian state-space model **separates** these effects, recovering the hidden degradation signal.

## Architecture

```
trackshift-tyre-intelligence/
├── app/                     # Streamlit dashboard
│   ├── dashboard.py         # Main entry point
│   ├── components.py        # UI components & charts
│   └── styling.py           # Dark motorsport theme
├── src/                     # Core library
│   ├── config.py            # All configurable parameters
│   ├── data/                # Ingestion, preprocessing, cache
│   ├── features/            # Fuel, tyre, temperature, track evolution
│   ├── models/              # Baselines, Bayesian SS, Kalman fallback
│   ├── analysis/            # Stint scoring, validation metrics
│   └── ev/                  # F1 → EV translation
├── data/                    # Data storage
│   ├── raw/                 # FastF1 cache
│   ├── processed/           # Processed DataFrames
│   └── demo/                # Bundled demo dataset
├── models/cached/           # Serialized model results
├── tests/                   # pytest suite (41 tests)
├── scripts/                 # Download, build, precompute
└── requirements.txt
```

## Data Sources

- **FastF1 API** — Official F1 timing data including lap times, tyre compounds, stint information, and weather data
- **Synthetic generator** — Deterministic fallback with known ground truth for validation

> ⚠️ **Fuel mass is estimated**, not measured. Exact race fuel loads are not publicly available.
> We use: `fuel_mass = 110 kg - 1.8 kg/lap × (lap_number - 1)`

## Installation

```bash
# Clone
git clone <repository-url>
cd trackshift-tyre-intelligence

# Install dependencies
pip install -r requirements.txt

# Build demo data (downloads FastF1 if available, falls back to synthetic)
python3 scripts/build_demo_dataset.py

# (Optional) Precompute models for instant demo
python3 scripts/precompute_model.py
```

## Running

```bash
# Launch the dashboard
bash run.sh

# Or directly:
streamlit run app/dashboard.py --theme.base dark --theme.primaryColor "#FF1744"
```

## Testing

```bash
python3 -m pytest tests/ -v
```

## Demo Mode

The application has **two data modes**:

| Mode | Source | Label |
|------|--------|-------|
| **Demo Mode** | Cached local data (real F1 or synthetic) | `DEMO DATA` or `SYNTHETIC VALIDATION DATASET` |
| **FastF1 Mode** | Live FastF1 API download | `REAL FASTF1 DATA` |

Demo Mode is the default — optimized for reliable demonstration without network dependency.

### 60-Second Demo Flow

1. **Launch** → TrackShift title appears
2. **Raw data** → Show lap time vs tyre age — car appears to get faster
3. **Confounders** → Reveal fuel mass declining, temperature changing
4. **Click "Isolate Tyre Degradation"** → Model runs
5. **Hero chart** → Latent degradation curve with uncertainty band appears
6. **KPIs** → Degradation rate, total degradation, confidence
7. **Model comparison** → Naive vs multivariate vs Bayesian
8. **F1 → EV** → Same abstraction for battery health

## Model Hierarchy

The system uses a **fallback hierarchy** to ensure reliability:

```
Bayesian State-Space (MCMC/NUTS)
    ↓ (if fails or unhealthy diagnostics)
Fast State-Space (Kalman MLE)
    ↓ (if fails)
Multivariate Regression (OLS)
```

The dashboard always indicates which model produced the displayed result.

## Limitations

This project acknowledges the following limitations:

1. **Fuel mass is estimated** — exact race fuel loads are proprietary
2. **Track evolution is approximated** — modeled as linear session progression
3. **Traffic detection is limited** — not all slow laps are identified
4. **No direct tyre measurements** — we infer *performance degradation*, not physical rubber loss
5. **Temperature effects are simplified** — tyre surface temp vs track temp vs air temp are related but distinct
6. **Compound differences** — inter-compound comparisons require additional modeling
7. **Driver behavior** — driving style changes are not explicitly modeled
8. **Uncertainty is posterior** — this is offline reconstruction, not real-time prediction

These limitations are **clearly documented** in the application UI.

## F1 → India: EV Battery Health

The same mathematical abstraction transfers directly to **electric vehicle battery State of Health (SoH) estimation**:

| F1 | EV |
|----|----|
| Tyre degradation (hidden) | Battery degradation / SoH (hidden) |
| Fuel mass (110 kg → 0 kg) | Payload (passengers/cargo) |
| Track temperature | Ambient temperature (Indian summers: 45°C+) |
| Lap performance | Energy consumption / range |
| Track evolution | Route / traffic conditions |

**Impact:** For a 5,000-vehicle Indian EV fleet at ₹3,00,000 per battery pack, accurately isolating true degradation from operational noise can defer ~₹15 Crore in annual replacement CapEx by eliminating premature battery replacements.

## References

- Cappello & Hoegh (2025), "A State-Space Approach to Modeling Tire Degradation in Formula 1", [arXiv:2512.00640](https://arxiv.org/abs/2512.00640)
- FastF1 Documentation: [https://docs.fastf1.dev](https://docs.fastf1.dev)
- FIA Technical Regulations: Fuel mass limit of 110 kg

---

*Built for the TrackShift Innovation Challenge 2026 — Plaksha University, Mphasis F1 Foundation, TGR Haas F1 Team*
