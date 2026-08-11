# Calibrated & Explainable Battery RUL Prediction

A computational study of uncertainty calibration for battery remaining useful life (RUL) prediction under distribution shift.

This project evaluates whether conformal prediction intervals maintain their target coverage when models are tested on unseen battery cells and across different battery datasets. SHAP is used to investigate which features drive model predictions and help explain why uncertainty coverage fails under distribution shift.

---

## Overview

Reliable uncertainty estimates are important for battery management systems, where a model may eventually be deployed on a battery cell that was not represented in its training data.

This project evaluates four generalization regimes:

* **Within-NASA:** prediction on an unseen NASA cell
* **Within-CALCE:** prediction on an unseen CALCE cell
* **NASA → CALCE:** training on NASA and testing on CALCE
* **CALCE → NASA:** training on CALCE and testing on NASA

For each regime, models are evaluated using leave-one-battery-out splits. Conformal prediction is used to construct adaptive prediction intervals, while SHAP is used to analyze feature reliance and investigate the behavior associated with coverage failures.

The project asks:

> **Can conformal prediction provide reliable RUL uncertainty intervals when a battery model is applied to cells or datasets that differ from those used for calibration?**

---

## Key Findings

Conformal prediction did not achieve its nominal **90% coverage target** in any of the evaluated regimes.

| Regime       | Evaluation                | Best Model | MAE (cycles) |    R² | Adaptive Coverage |
| ------------ | ------------------------- | ---------- | -----------: | ----: | ----------------: |
| Within-NASA  | Unseen cell, same dataset | XGBoost    |         22.7 |  0.32 |             22.4% |
| Within-CALCE | Unseen cell, same dataset | XGBoost    |         83.4 |  0.26 |             15.5% |
| NASA → CALCE | Unseen dataset            | ElasticNet |         88.4 |  0.33 |             13.9% |
| CALCE → NASA | Unseen dataset            | XGBoost    |        305.2 | -64.3 |              0.0% |

### 1. Coverage fails even within the same dataset

Adaptive conformal intervals achieved only **15–32% empirical coverage** in the within-dataset experiments despite a nominal target of 90%.

This suggests that the assumptions required for conformal validity can be violated when moving between battery cells, even when those cells belong to the same dataset.

### 2. Cross-dataset distribution shift is substantially worse

Performance deteriorates further when models are transferred between NASA and CALCE.

The degradation is strongly directional:

* **NASA → CALCE** remains poor but produces positive R² for the best model.
* **CALCE → NASA** produces an R² of **-64.3** for the best model and **0% adaptive coverage**.

### 3. SHAP reveals a feature-reliance shift

SHAP analysis indicates that models trained on NASA and CALCE rely on substantially different features.

Models trained on CALCE rely heavily on `global_cycle`, with SHAP magnitudes of approximately **80–127**.

Models trained on NASA rely more heavily on `avg_voltage`, with SHAP magnitudes of approximately **25–49**.

The datasets also occupy very different cycle-count ranges:

* **CALCE:** approximately 300–550 cycles
* **NASA:** approximately 110–170 cycles

This creates a strong distribution shift in the relationship between cycle count and RUL. Tree-based models are particularly vulnerable because they cannot extrapolate beyond the cycle-count ranges represented during training.

Together, the model family, feature reliance, and direction of distribution shift provide a mechanistic explanation for the observed coverage failures.

---

## Methodology

The pipeline evaluates three model families:

* **ElasticNet**
* **Random Forest**
* **XGBoost**

The complete pipeline consists of:

1. Parsing the raw NASA PCoE and CALCE battery datasets
2. Cleaning and reconstructing per-cycle measurements
3. Engineering common battery-health features
4. Generating threshold-based RUL labels
5. Training models using leave-one-battery-out evaluation
6. Generating adaptive conformal prediction intervals
7. Measuring empirical coverage and prediction error
8. Computing SHAP feature attributions
9. Comparing feature reliance and failure behavior across datasets

---

## Datasets

### NASA PCoE

The NASA Prognostics Center of Excellence battery dataset provides `.mat` files for cells including:

* B0005
* B0006
* B0007
* B0018

NASA RUL labels follow the dataset's **70%-of-reference-capacity** convention, corresponding to 30% capacity fade.

### CALCE CS2

The CALCE dataset uses Arbin `.xlsx` logs for:

* CS2_35
* CS2_36
* CS2_37
* CS2_38

CALCE RUL labels use an **80%-of-reference-capacity** end-of-life threshold.

---

## Data Processing

Several preprocessing steps are required to obtain reliable cycle-level measurements from the raw data.

### CALCE

The `Discharge_Capacity(Ah)` field is cumulative within each dated Arbin log rather than directly representing per-cycle capacity. Per-cycle capacity is therefore reconstructed through differencing.

The preprocessing also removes:

* Partial final cycles from individual log files
* Short or interrupted check cycles embedded in the test schedule
* Other artifacts that would otherwise appear as artificial capacity degradation

These operations are implemented in:

```text
src/features_calce.py
```

### RUL Labeling

RUL is defined using capacity-based end-of-life thresholds rather than simply counting cycles until the end of the available test data.

Three of the four NASA cells do not reach the specified threshold before testing ends. These labels are therefore right-censored and are explicitly flagged in the processing pipeline.

---

## How to Use

The project is designed to be run through a single entry point.

### 1. Install the dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

### 2. Run the complete pipeline

```bash
python run_all.py
```

The pipeline automatically runs the data processing, model evaluation, conformal prediction analysis, SHAP analysis, and figure generation.

The main outputs are written to the `results/` directory.

### Quick Start

```bash
git clone <this-repo-url>
cd battery-rul
pip install -r requirements.txt
python run_all.py
```

When using the processed dataset included in the repository, no additional commands are required to reproduce the main experiments.

---

## Reproducibility

The repository includes a processed dataset so that the main experiments can be reproduced without downloading the original raw datasets.

Running:

```bash
python run_all.py
```

will generate the experiment results and figures in:

```text
results/
```

### Rebuilding From Raw Data

To rebuild the processed dataset from the original sources, place the NASA and CALCE raw files in:

```text
data/raw/nasa/fy08q4/
data/raw/calce/CS2_XX/
```

The pipeline automatically detects the raw data and rebuilds the processed dataset before running the experiments.

---

## Results

The primary summary results are stored in:

```text
results/v2_summary.csv
```

Additional experiment outputs include:

```text
results/v2_results.csv
results/v2_shap.csv
```

### Figures

The primary coverage figure is:

```text
results/coverage_by_regime.png
```

The SHAP feature-reliance analysis is visualized in:

```text
results/shap_feature_flip.png
```

The processed CALCE state-of-health curves are available in:

```text
results/calce_soh_curves.png
```

---

## Repository Structure

```text
battery-rul/
├── run_all.py
├── run_experiment_v2.py
├── requirements.txt
│
├── src/
│   ├── parse_calce.py
│   ├── features_calce.py
│   ├── features_nasa.py
│   ├── build_dataset.py
│   ├── models.py
│   ├── conformal.py
│   └── make_figures.py
│
├── results/
│   ├── combined_dataset.csv
│   ├── v2_summary.csv
│   ├── v2_results.csv
│   ├── v2_shap.csv
│   ├── coverage_by_regime.png
│   ├── shap_feature_flip.png
│   └── calce_soh_curves.png
│
├── paper/
│   └── main.tex
│
└── data/
    └── raw/
```

---

## Source Files

| File                | Description                                                       |
| ------------------- | ----------------------------------------------------------------- |
| `parse_calce.py`    | Parses raw CALCE Arbin `.xlsx` logs                               |
| `features_calce.py` | CALCE feature engineering, RUL labeling, and artifact filtering   |
| `features_nasa.py`  | NASA feature engineering and RUL labeling                         |
| `build_dataset.py`  | Combines NASA and CALCE data into a common feature representation |
| `models.py`         | ElasticNet, Random Forest, and XGBoost models                     |
| `conformal.py`      | Conformal prediction and coverage analysis                        |
| `make_figures.py`   | Generates the project figures                                     |
| `run_all.py`        | Runs the complete project pipeline                                |

---

## Limitations

The current evaluation has a relatively small number of battery cells, particularly within the individual datasets. As a result, the reported coverage estimates can have substantial statistical uncertainty.

The NASA dataset also contains cells that do not reach the selected end-of-life threshold before testing ends, resulting in right-censored RUL labels.

The current analysis focuses on cross-cell and cross-dataset distribution shift. Additional datasets and larger cell populations would provide a stronger evaluation of the generality of the findings.

The results should therefore be interpreted as a computational study of conformal uncertainty behavior under battery-level distribution shift rather than as a definitive characterization of conformal prediction across all battery datasets.

---

## Research Scope

The central research question is:

> **Can conformal prediction provide reliable RUL uncertainty intervals when a battery model is applied to cells or datasets that differ from those used for calibration?**

The results indicate that nominal conformal coverage can break down substantially under realistic battery-level distribution shift.

The SHAP analysis further suggests that these failures are associated with changes in feature reliance and dataset-specific relationships between battery cycle information and RUL.

---

## License

License information will be added here.

---

## Citation

A citation will be added when the associated research paper or preprint is publicly available.
