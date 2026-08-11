"""
make_figures.py — Regenerate the two headline figures from results/v2_summary.csv
and results/v2_shap.csv. Run after run_experiment_v2.py (run_all.py does both).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results"

REGIMES = ["within_NASA", "within_CALCE", "NASA_to_CALCE", "CALCE_to_NASA"]
REGIME_LABELS = ["Within-NASA\n(same product)", "Within-CALCE\n(same product)",
                  "NASA train\n->CALCE test", "CALCE train\n->NASA test"]
MODELS = ["ElasticNet", "RandomForest", "XGBoost"]

FEATURES = ["global_cycle", "avg_voltage", "discharge_time_s", "voltage_drop", "avg_current"]
FEAT_LABELS = ["cycle #", "avg voltage", "discharge time", "voltage drop", "avg current"]


def coverage_by_regime_figure():
    df = pd.read_csv(f"{RESULTS_DIR}/v2_summary.csv")
    x = np.arange(len(REGIMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(MODELS):
        vals = []
        for r in REGIMES:
            row = df[(df.regime == r) & (df.model == m)]
            vals.append(row["adaptive_coverage"].values[0] * 100 if len(row) else 0)
        ax.bar(x + (i - 1) * width, vals, width, label=m)
    ax.axhline(90, color="red", linestyle="--", linewidth=1, label="90% target")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylabel("Empirical conformal coverage (%)")
    ax.set_title("Conformal coverage collapses with distribution shift severity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/coverage_by_regime.png", dpi=130)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/coverage_by_regime.png")


def shap_feature_flip_figure():
    shap_df = pd.read_csv(f"{RESULTS_DIR}/v2_shap.csv")
    top = shap_df.groupby(["regime", "model", "feature"])["mean_abs_shap"].mean().reset_index()
    xgb = top[top.model == "XGBoost"]

    data = np.zeros((len(FEATURES), len(REGIMES)))
    for i, f in enumerate(FEATURES):
        for j, r in enumerate(REGIMES):
            row = xgb[(xgb.feature == f) & (xgb.regime == r)]
            data[i, j] = row["mean_abs_shap"].values[0] if len(row) else 0
    data_norm = data / data.sum(axis=0, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(REGIMES))
    bottom = np.zeros(len(REGIMES))
    colors = plt.cm.tab10(np.linspace(0, 1, len(FEATURES)))
    for i, f in enumerate(FEAT_LABELS):
        ax.bar(x, data_norm[i], bottom=bottom, label=f, color=colors[i])
        bottom += data_norm[i]
    ax.set_xticks(x)
    ax.set_xticklabels(["Within-NASA", "Within-CALCE", "NASA->CALCE", "CALCE->NASA"])
    ax.set_ylabel("Share of total |SHAP| attribution")
    ax.set_title("XGBoost: which feature the model trusts flips with training domain")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/shap_feature_flip.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/shap_feature_flip.png")


def make_all_figures():
    coverage_by_regime_figure()
    shap_feature_flip_figure()


if __name__ == "__main__":
    make_all_figures()
