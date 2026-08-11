"""
run_all.py — ONE command to reproduce everything in results/.

    pip install -r requirements.txt
    python run_all.py

Runs the leave-one-battery-out + cross-dataset conformal prediction + SHAP
experiment (run_experiment_v2.py) and regenerates both headline figures
(src/make_figures.py). Takes well under a minute on a laptop.

Works with zero setup: this repo ships the already-processed
results/combined_dataset.csv, so you get results immediately even without
the raw NASA/CALCE data. If you place the raw data under data/raw/ (see
README.md), it will use that instead and rebuild the dataset from scratch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment_v2 import run as run_experiment
from src.make_figures import make_all_figures


def main():
    print("=" * 70)
    print("Battery RUL: conformal prediction + SHAP cross-dataset stress test")
    print("=" * 70)

    results_df, shap_df = run_experiment()

    print("\nRegenerating figures...")
    make_all_figures()

    print("\nDone. See:")
    print("  results/v2_summary.csv           - headline results table")
    print("  results/v2_results.csv           - per-battery detail")
    print("  results/v2_shap.csv              - SHAP attributions")
    print("  results/coverage_by_regime.png   - headline figure")
    print("  results/shap_feature_flip.png    - mechanistic SHAP figure")


if __name__ == "__main__":
    main()
