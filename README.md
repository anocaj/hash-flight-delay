# HASH Machine Learning Challenge: Flight Delay Prediction

## Quick Start for Reviewers

**Main Submission**: Open `notebooks/01_flight_delay_analysis.ipynb` and view all cells. If necessary rerun all cells (5-15 min runtime)

**Key Result**: 0.77 AUC-ROC predicting flight delays using only pre-departure information (11% improvement over 0.66 baseline)

---

## Project Summary

### Objective

Predict whether a US domestic flight will be delayed ≥15 minutes (`ARR_DEL15`) using only information available **before departure**.

### Approach

Initial exploration (`initial_exploration.ipynb`) showed that including `DEP_DELAY` (departure delay) yields >90% AUC—impressive but unrealistic, as this information is only available after takeoff. Without departure features, baseline performance drops to ~66% AUC. The full analysis is documented in the main notebook, `01_flight_delay_analysis.ipynb`.

Realistic Pre-Departure Analysis: To enable actionable predictions, I engineered 70+ features available before departure—weather, temporal patterns, aircraft and carrier history—and achieved 0.77 AUC.

### Key Findings

**Model Performance**:

- AUC-ROC: 0.77 (solid for pre-departure prediction)
- Precision/Recall: 0.65/0.24
- Model: LightGBM (outperformed Random Forest and Logistic Regression)

**Top Predictive Features**:

![Flight Delay Cause-Effect Diagram](assets/cause-effect-diagram.png)

_Fishbone diagram showing the six major categories driving flight delays: Weather (storms, wind, precipitation), Scheduling (peak hours, tight connections), External Factors (COVID, strikes, holidays), Aircraft & Maintenance (mechanical issues, aircraft swaps), Airport & Air Traffic (congestion, ATC delays), and Airline Operations (previous flight delays, crew scheduling, gate turnaround)._

1. Aircraft delay history (previous flight delays for same tail number)
2. Temporal patterns (time of day, day of week)
3. Weather conditions (temperature, wind, precipitation)
4. Carrier performance (historical on-time rates)
5. Route characteristics (distance, airport congestion)

**Business Value**:

- Enable proactive passenger rebooking
- Optimize crew scheduling and resource allocation
- Reduce cascading delay effects

---

## Repository Structure

```
hash-flight-delay/
├── notebooks/
│   ├── 01_flight_delay_analysis.ipynb    ⭐ MAIN SUBMISSION (67 cells)
│   └── initial_exploration.ipynb          📊 Exploration work
│
├── src/                                   🐍 Production modules
│   ├── data/data_prep.py                  # Data loading, weather API
│   ├── features/feature_engineering.py    # 70+ feature creation
│   ├── models/train_model.py              # LightGBM, RF, LogReg
│   ├── models/evaluate.py                 # Performance metrics
│   └── visualization/plots.py             # All plotting functions
│   └──  outputs/                          # Generated outputs
│
├── data/
│   └── raw/
│       ├── flights_jan_2025.csv
│       ├── flights_feb_2025.csv
│       └── airports.csv
```

---

## Setup & Execution

### Prerequisites

- Python 3.8+
- Jupyter Notebook

### Installation

```bash
# Navigate to project
cd hash-flight-delay

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Launch main submission notebook
jupyter notebook notebooks/01_flight_delay_analysis.ipynb

# Run all cells: Kernel → Restart & Run All
# Runtime: 5-20 minutes
```

---

## Technical Approach

### Data Leakage Prevention

**Excluded** (only known post-departure):

- `DEP_DELAY`, `DEP_DEL15` - Departure delay info
- `ARR_DELAY` - Target-related
- `ACTUAL_ELAPSED_TIME`, `AIR_TIME` - Post-flight

**Included** (known pre-departure):

- Scheduled times, historical performance, weather forecasts, temporal patterns

### Methodology

- **Split**: 60/20/20 train/val/test with stratification
- **Models Tested**: LightGBM (best), Random Forest, Logistic Regression
- **Hyperparameter Tuning**: Optuna (5 trials - limited by 6-hour constraint)
- **Evaluation**: AUC-ROC, Precision, Recall, F1-Score

---

## Key Limitations

1. **Temporal Scope**: Trained only on Jan-Feb 2025 (winter months) - missing summer patterns
2. **Class Imbalance**: 80/20 on-time/delayed distribution - model favors majority class
3. **Hyperparameter Tuning**: Only 5 trials (see Next Steps in notebook for expansion plan)
4. **Missing Factors**: ATC delays, crew scheduling, gate availability, special events
5. **Validation Strategy**: Random split instead of time-based (noted in limitations section)

---

## Next Steps (Beyond 6-Hour Scope)

As documented in the notebook:

1. **Hyperparameter Tuning** (Priority 1): Increase from 5→50-100 Optuna trials (~0.02-0.03 AUC gain)
2. **Temporal Validation**: Time-based train/test split for robust evaluation
3. **Error Analysis**: Analyze misclassifications by carrier/route/time
4. **Model Deployment**: Save models, create inference pipeline, implement drift monitoring

---

## Dependencies

**Core**: pandas, numpy, scikit-learn, lightgbm, optuna (tuning)
**Visualization**: matplotlib, seaborn, shap
**External Data**: meteostat (weather API), Aircraft Registration registry.faa.gov (https://registry.faa.gov/), airports (https://ourairports.com/data/)
**Development**: jupyter

Full list in `requirements.txt`

---

## Challenge Deliverables

✅ **Feature Importance & Results** - SHAP analysis, top drivers identified
✅ **Model Approach & Rationale** - Systematic comparison of 3 algorithms
✅ **Limitations & Bias Analysis** - Honest assessment with mitigation strategies

All deliverables in `notebooks/01_flight_delay_analysis.ipynb` with clear documentation and business context.

---

**Time Constraint**: Analysis completed within 6-hour limit - hyperparameter tuning and advanced validation deferred to "Next Steps"
