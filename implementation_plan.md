# TrackShift Tyre Degradation Intelligence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, polished prototype that demonstrates how F1 tyre degradation can be estimated as a latent hidden state after accounting for confounders (fuel, temperature, track evolution) — showing that raw lap times are misleading.

**Architecture:** Modular Python application with data ingestion (FastF1 + offline fallback), feature engineering (fuel/temp/track), three-tier model hierarchy (naive → multivariate → Bayesian state-space), and a Streamlit dashboard that tells the "peeling away confounders" story. All model results are cached for instant demo playback.

**Tech Stack:** Python 3.12, FastF1, pandas, numpy, scipy, scikit-learn, PyMC, ArviZ, Plotly, Streamlit, pydantic, joblib

**Spec:** User request (inline above)

## Global Constraints

- Python 3.12 compatible
- No cloud infrastructure / microservices / databases
- No fabricated F1 results labeled as real
- All scientific assumptions documented
- Prototype must work fully offline with cached/demo data
- Every chart answers a question — no chart spam
- `pytest` for all tests
- Streamlit for frontend — dark motorsport aesthetic
- Plotly for all visualizations

---

## Environment Status

| Component | Status |
|-----------|--------|
| Python 3.12.2 | ✅ Installed |
| pandas 2.2.2 | ✅ Installed |
| numpy 2.4.4 | ✅ Installed |
| scipy 1.17.1 | ✅ Installed |
| scikit-learn 1.5.2 | ✅ Installed |
| plotly 6.9.0 | ✅ Installed |
| streamlit 1.61.1 | ✅ Installed |
| pydantic 2.12.5 | ✅ Installed |
| joblib 1.4.2 | ✅ Installed |
| FastF1 | 🔄 Installing |
| PyMC | 🔄 Installing |
| ArviZ | 🔄 Installing |

---

## File Structure

```
trackshift-tyre-intelligence/
├── app/
│   ├── dashboard.py           # Main Streamlit entry point
│   ├── components.py          # Reusable UI components (KPI cards, sections)
│   └── styling.py             # CSS, theme, color constants
│
├── src/
│   ├── __init__.py
│   ├── config.py              # All configuration (fuel, sampling, paths)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fastf1_loader.py   # FastF1 session loading with caching
│   │   ├── preprocessing.py   # Lap filtering, cleaning, summary
│   │   ├── cache.py           # Local data cache management
│   │   └── demo_data.py       # Synthetic data generator + bundled demo
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── fuel.py            # Fuel mass estimation
│   │   ├── tyre_features.py   # Tyre age, stint extraction
│   │   └── confounders.py     # Temperature joining, track evolution
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py        # Naive + multivariate regression
│   │   ├── state_space.py     # Bayesian state-space model (PyMC)
│   │   ├── fast_fallback.py   # Kalman/MAP fallback
│   │   ├── inference.py       # Model runner with fallback hierarchy
│   │   └── diagnostics.py     # R-hat, ESS, divergence checks
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── stint_analysis.py  # Stint quality scoring, demo selection
│   │   └── metrics.py         # MAE, RMSE, temporal validation
│   │
│   └── ev/
│       ├── __init__.py
│       └── translation.py     # F1→EV analogy content
│
├── data/
│   ├── raw/                   # Raw FastF1 cache
│   ├── processed/             # Processed stint DataFrames
│   └── demo/                  # Bundled demo dataset + mock EV data
│
├── models/
│   └── cached/                # Serialized model results
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_baseline.py
│   ├── test_state_space.py
│   └── test_demo_data.py
│
├── scripts/
│   ├── download_fastf1.py     # One-time FastF1 download
│   ├── build_demo_dataset.py  # Generate bundled demo data
│   └── precompute_model.py    # Pre-run models for cached results
│
├── requirements.txt
├── README.md
├── .gitignore
└── run.sh
```

---

## Proposed Changes

### Task 1: Project Scaffolding & Configuration

Create the entire directory structure, `requirements.txt`, `.gitignore`, `run.sh`, and the central `config.py`.

#### [NEW] [requirements.txt](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/requirements.txt)
Pin all dependencies with compatible ranges.

#### [NEW] [.gitignore](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/.gitignore)
Standard Python gitignore + data/raw, models/cached, fastf1 cache.

#### [NEW] [run.sh](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/run.sh)
Single command: `streamlit run app/dashboard.py`

#### [NEW] [src/config.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/config.py)
- Pydantic settings model with all configurable parameters
- Fuel: `INITIAL_FUEL_KG=110`, `FUEL_BURN_PER_LAP_KG=1.8`
- Sampling: `DRAWS=500`, `TUNE=500`, `CHAINS=2`
- Paths: data dirs, cache dirs
- Default race: `2024 Italian Grand Prix` (Monza — long straights, clear deg)

**Interfaces:**
- Produces: `Config` pydantic model used by all modules

---

### Task 2: Data Ingestion — FastF1 Loader

#### [NEW] [src/data/fastf1_loader.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/data/fastf1_loader.py)
- `load_session(year, gp, session_type)` → loads laps + weather
- FastF1 cache enabled under `data/raw/`
- Graceful failure: returns `None` + error message if network/API fails
- No telemetry loading (lap-level only)

#### [NEW] [src/data/cache.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/data/cache.py)
- `save_processed(df, key)` / `load_processed(key)` — pickle/parquet
- `cache_key(race, driver, stint, config_hash)` — deterministic key
- `save_model_result(result, key)` / `load_model_result(key)`

**Interfaces:**
- Produces: `load_session()` → `(laps_df, weather_df)` or `(None, None)`

---

### Task 3: Data Cleaning & Preprocessing

#### [NEW] [src/data/preprocessing.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/data/preprocessing.py)
- `clean_laps(laps_df)` → filtered DataFrame + `PreprocessingSummary`
- Filters: missing LapTime, pit in/out laps, safety car, VSC, statistical outliers (IQR-based), invalid timing
- `PreprocessingSummary` dataclass: raw_count, valid_count, removed_count, removal_reasons dict
- Every filter step logged with reason

**Interfaces:**
- Consumes: `laps_df` from `fastf1_loader.load_session()`
- Produces: `(clean_df, PreprocessingSummary)`

---

### Task 4: Stint Extraction & Tyre Features

#### [NEW] [src/features/tyre_features.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/features/tyre_features.py)
- `extract_stints(clean_laps_df, driver)` → list of `StintInfo` dataclasses
- `StintInfo`: driver, compound, stint_number, start_lap, end_lap, num_laps, tyre_ages, lap_times
- Uses FastF1's `TyreLife` field if available, otherwise derives from stint boundaries
- `get_stint_df(stints, stint_idx)` → DataFrame with all stint data

**Interfaces:**
- Consumes: `clean_df` from preprocessing
- Produces: `list[StintInfo]`, stint DataFrames with `tyre_age` column

---

### Task 5: Confounder Feature Engineering

#### [NEW] [src/features/fuel.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/features/fuel.py)
- `estimate_fuel_mass(stint_df, config)` → adds `fuel_mass_kg` column
- `fuel_mass = initial_fuel - burn_rate * race_progress`
- Race progress from absolute lap number

#### [NEW] [src/features/confounders.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/features/confounders.py)
- `join_weather(stint_df, weather_df)` → adds `track_temp_C`, `air_temp_C`
- Nearest-timestamp matching (no look-ahead)
- `add_track_evolution(stint_df)` → adds `track_progress` (normalized 0-1)
- `build_feature_matrix(stint_df, config)` → orchestrates all features

**Interfaces:**
- Consumes: stint DataFrame, weather DataFrame, Config
- Produces: fully-featured DataFrame with columns: `lap_time_seconds`, `tyre_age`, `fuel_mass_kg`, `track_temp_C`, `track_progress`

---

### Task 6: Synthetic Demo Data Generator

#### [NEW] [src/data/demo_data.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/data/demo_data.py)
- `generate_synthetic_stint(seed, config)` → DataFrame with known ground truth
- Ground truth degradation: `base_rate * tyre_age + nonlinear_term`
- Generates: `lap_time = base + degradation + fuel_effect + temp_effect + track_effect + noise`
- Includes occasional positive outliers (lockups/traffic)
- `load_demo_dataset()` → loads bundled pre-generated demo data from `data/demo/`
- Clearly labeled `is_synthetic=True`

#### [NEW] [scripts/build_demo_dataset.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/scripts/build_demo_dataset.py)
- Downloads FastF1 data for default race, preprocesses, saves to `data/demo/`
- Falls back to synthetic generation if FastF1 unavailable

**Interfaces:**
- Produces: `demo_stint_df` with all features + `true_degradation` column (synthetic only)

---

### Task 7: Baseline Models

#### [NEW] [src/models/baseline.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/models/baseline.py)

**Baseline 1 — Naive regression:**
- `fit_naive(stint_df)` → `NaiveResult`
- `LapTime = α + β·TyreAge + ε`
- Returns: coefficients, predictions, residuals, R²

**Baseline 2 — Multivariate regression:**
- `fit_multivariate(stint_df)` → `MultivariateResult`
- `LapTime = α + β_tyre·TyreAge + β_fuel·Fuel + β_temp·Temp + β_progress·Progress + ε`
- Returns: coefficients, predictions, residuals, R², feature importances

Both use scikit-learn `LinearRegression`.

**Interfaces:**
- Consumes: featured stint DataFrame
- Produces: `NaiveResult`, `MultivariateResult` dataclasses

---

### Task 8: Bayesian State-Space Model

#### [NEW] [src/models/state_space.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/models/state_space.py)

Core model:
```
y_t = β₀ + β_f·f_t + β_c·c_t + β_p·p_t + α_t + ε_t
α_t = α_{t-1} + δ + ω_t
```

- `build_model(stint_df, config)` → PyMC model
- `fit_model(model, config)` → `BayesianResult`
- Priors: weakly informative, tuned to data scale
  - `β₀ ~ Normal(median_laptime, 5.0)`
  - `δ ~ HalfNormal(0.1)` (degradation drift ≥ 0)
  - `β_f ~ Normal(0, 1.0)`
  - `β_c ~ Normal(0, 0.5)`
  - `β_p ~ Normal(0, 2.0)`
  - `σ_process ~ HalfNormal(0.1)`
  - `σ_obs ~ HalfNormal(1.0)`
  - `ν ~ Gamma(2, 0.1)` (Student-T df)
- Likelihood: Student-T for robustness
- Returns: posterior degradation curve (mean, median, 5th/95th percentiles), rate, diagnostics

**Interfaces:**
- Consumes: featured stint DataFrame, Config
- Produces: `BayesianResult` with `degradation_mean`, `degradation_p05`, `degradation_p95`, `rate_per_lap`, `total_degradation`, `idata` (ArviZ InferenceData)

---

### Task 9: Fast Fallback Model

#### [NEW] [src/models/fast_fallback.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/models/fast_fallback.py)
- Kalman-filter-based state-space estimation (no MCMC)
- `fit_kalman(stint_df, config)` → `KalmanResult`
- Uses scipy's optimization or manual Kalman recursion
- Produces degradation curve + approximate uncertainty
- Much faster than MCMC — used when PyMC fails or is too slow

**Interfaces:**
- Produces: `KalmanResult` compatible with `BayesianResult` interface

---

### Task 10: Model Inference Orchestrator & Diagnostics

#### [NEW] [src/models/inference.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/models/inference.py)
- `run_degradation_analysis(stint_df, config, model_choice)` → orchestrates model fitting
- Fallback hierarchy: Bayesian → Kalman → Multivariate
- Catches PyMC errors, checks diagnostics, falls back if needed
- Caches results via `cache.py`

#### [NEW] [src/models/diagnostics.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/models/diagnostics.py)
- `check_diagnostics(idata)` → `DiagnosticsSummary`
- Checks: R-hat, ESS, divergences
- `is_healthy(summary)` → bool
- Threshold: R-hat < 1.01, ESS > 100, divergences < 5%

---

### Task 11: Analysis — Stint Selection & Validation Metrics

#### [NEW] [src/analysis/stint_analysis.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/analysis/stint_analysis.py)
- `score_stint(stint_info, stint_df)` → quality float
- Factors: length (≥15 laps), valid lap ratio, data completeness, tyre-age range, apparent degradation signal
- `select_best_stint(all_stints)` → best `StintInfo`
- `auto_select_demo(session_data)` → fully-selected driver+stint for demo mode

#### [NEW] [src/analysis/metrics.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/analysis/metrics.py)
- `compute_metrics(actual, predicted)` → MAE, RMSE, correlation
- `temporal_validation(stint_df, model_fn, train_frac=0.7)` → train/test metrics
- Temporal split only (no random shuffle)

---

### Task 12: EV Translation Module

#### [NEW] [src/ev/translation.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/src/ev/translation.py)
- `get_ev_analogy_data()` → dictionary of F1→EV mappings
- `generate_mock_ev_data()` → simple EV telemetry DataFrame
- Content for the "Motorsport → India" section

#### [NEW] [data/demo/mock_ev_telemetry.csv](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/data/demo/mock_ev_telemetry.csv)
Mock EV battery data with columns: vehicle_id, trip_id, distance_km, payload_kg, ambient_temp_C, energy_used_kWh, voltage, current, estimated_battery_health

---

### Task 13: Streamlit Dashboard — Styling & Components

#### [NEW] [app/styling.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/app/styling.py)
- Dark motorsport theme CSS
- Color palette: dark bg (#0E1117), accent red (#FF1744), tyre compound colors
- Plotly template with dark background, minimal grid
- Typography consistent across all charts

#### [NEW] [app/components.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/app/components.py)
- `render_kpi_card(title, value, unit, delta=None)` → styled metric card
- `render_section_header(title, subtitle)` → styled section
- `render_data_quality(summary)` → expandable data quality
- `render_diagnostics(diagnostics)` → expandable model diagnostics
- `render_model_comparison_table(naive, multi, bayesian)` → comparison
- `render_engineering_insight(results)` → natural language summary
- `render_ev_section()` → F1→EV translation section
- Chart builders for all 7 key visualizations using Plotly

---

### Task 14: Streamlit Dashboard — Main App

#### [NEW] [app/dashboard.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/app/dashboard.py)

Full dashboard implementing all 8 sections from spec:

1. **Header** — Title, status indicators
2. **KPI Row** — Degradation rate, total deg, stint length, confidence
3. **Section 1: The Paradox** — Raw lap time vs tyre age with naive regression
4. **Section 2: Confounder Layer** — Fuel, temperature, track evolution charts
5. **Section 3: Degradation Engine** — Hero chart with posterior + credible interval
6. **Section 4: Model Comparison** — Table comparing 3 models
7. **Section 5: Engineering Insight** — Generated natural-language summary
8. **Section 6: Data Quality** — Expandable preprocessing summary
9. **Section 7: Model Diagnostics** — Expandable R-hat, ESS, divergences
10. **How It Works** — Model explanation
11. **EV Translation** — F1→India section

**Sidebar controls:**
- Data source: Demo Mode / FastF1
- Race dropdown (if FastF1)
- Driver dropdown
- Stint dropdown
- Model choice
- Advanced settings (expandable): fuel params, sampling config
- "RUN DEGRADATION ANALYSIS" button

**Demo Mode flow:**
1. Auto-loads best cached stint
2. Shows raw paradox
3. "Reveal True Degradation" button
4. Transitions to degradation view

**Caching:** `@st.cache_data` and `@st.cache_resource` for all expensive operations

---

### Task 15: Scripts — Download, Build Demo, Precompute

#### [NEW] [scripts/download_fastf1.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/scripts/download_fastf1.py)
- Downloads specified race sessions via FastF1
- Saves to `data/raw/` cache

#### [NEW] [scripts/build_demo_dataset.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/scripts/build_demo_dataset.py)
- Processes FastF1 data → saves to `data/demo/`
- Falls back to synthetic if FastF1 unavailable

#### [NEW] [scripts/precompute_model.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/scripts/precompute_model.py)
- Runs all 3 models on demo data
- Saves results to `models/cached/`

---

### Task 16: Tests

#### [NEW] [tests/test_preprocessing.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/tests/test_preprocessing.py)
- Test lap filtering (pit laps removed, outliers detected)
- Test preprocessing summary counts

#### [NEW] [tests/test_features.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/tests/test_features.py)
- Test tyre age calculation
- Test fuel mass estimation
- Test weather joining
- Test track evolution feature

#### [NEW] [tests/test_baseline.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/tests/test_baseline.py)
- Test naive model on synthetic data
- Test multivariate model on synthetic data
- Test coefficient signs

#### [NEW] [tests/test_state_space.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/tests/test_state_space.py)
- Test model builds without error
- Test on synthetic data with known ground truth
- Test degradation normalization

#### [NEW] [tests/test_demo_data.py](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/tests/test_demo_data.py)
- Test synthetic data has correct columns
- Test ground truth degradation is positive
- Test outliers present

---

### Task 17: README & Documentation

#### [NEW] [README.md](file:///Users/pushpitjain/Documents/antigravity/modest-salk/trackshift-tyre-intelligence/README.md)
- Full professional README per spec section 37-39
- Mathematical model documentation
- Installation & run instructions
- Limitations section
- F1→EV abstraction

---

## Verification Plan

### Automated Tests
```bash
cd trackshift-tyre-intelligence && python -m pytest tests/ -v
```

### Manual Verification
1. Run `bash run.sh` — Streamlit launches without errors
2. Demo Mode auto-loads and displays paradox chart
3. Click "Reveal True Degradation" — model runs, degradation curve appears
4. Switch to FastF1 mode — data loads (if network available)
5. Select different driver/stint — charts update
6. Verify no tracebacks in any interaction
7. Verify all 7 key charts render correctly
8. Verify model comparison table shows all 3 models
9. Verify engineering insight generates with real numbers

### Performance Targets
- Demo mode startup: < 10 seconds (with cached data)
- Model execution: < 15 seconds (Bayesian), < 2 seconds (Kalman fallback)

---

## Open Questions

> [!IMPORTANT]
> **Default Race Selection:** I plan to use the **2024 Italian GP (Monza)** as the default demo race. Monza typically has clear degradation patterns with long stints. Should I use a different race? Other good candidates: 2024 Spanish GP, 2023 Bahrain GP.

> [!IMPORTANT]
> **PyMC Sampling Budget:** For demo responsiveness, I plan to use `draws=500, tune=500, chains=2` (total ~2000 posterior samples). This trades some posterior quality for speed (~10-15 seconds). Is this acceptable, or should I prioritize more thorough sampling with longer wait times?

> [!NOTE]
> **EV Section Depth:** Per spec, the EV translation is lightweight/secondary. I'll implement it as a static comparison section with mock data visualization — no actual battery degradation model. Confirm this is the right scope.
