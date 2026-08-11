"""
run_experiment_v2.py — Real experiment on properly parsed raw NASA + CALCE
data (replaces run_experiment.py's prototype-CSV version).

Three evaluation regimes, all leave-one-battery-out at the row level
(never split cycles from the same cell across train/test):
  1. WITHIN-NASA:  train on 3 NASA cells,  test on the 4th held-out NASA cell
  2. WITHIN-CALCE: train on 3 CALCE cells, test on the 4th held-out CALCE cell
  3. CROSS-DATASET: train on ALL of one dataset, test on EACH cell of the
     other (both directions) -- the real distribution-shift stress test.

Same three model families + same two conformal flavors (standard, locally
adaptive) + SHAP as the original prototype, so results are apples-to-apples
comparable to what we already reported.
"""

import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

from src.build_dataset import load_all, COMMON_FEATURES, TARGET
from src.models import build_models
from src.conformal import (
    split_conformal_intervals,
    normalized_conformal_intervals,
    empirical_coverage,
    mean_interval_width,
)

ALPHA = 0.1
RESULTS_DIR = "results"


def make_split(df, train_ids, cal_ids, test_id, seed=0):
    train_df = df[df["battery_id"].isin(train_ids)]
    cal_df = df[df["battery_id"].isin(cal_ids)] if cal_ids else None
    test_df = df[df["battery_id"] == test_id]

    if cal_df is None:
        # carve calibration out of train at the row level
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(train_df))
        n_cal = int(len(train_df) * 0.25)
        cal_df = train_df.iloc[idx[:n_cal]]
        train_df = train_df.iloc[idx[n_cal:]]

    X_train, y_train = train_df[COMMON_FEATURES].values, train_df[TARGET].values
    X_cal, y_cal = cal_df[COMMON_FEATURES].values, cal_df[TARGET].values
    X_test, y_test = test_df[COMMON_FEATURES].values, test_df[TARGET].values
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def evaluate(regime, test_id, X_train, y_train, X_cal, y_cal, X_test, y_test, rows, shap_records, seed=0):
    models = build_models(seed=seed)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        yhat_cal = model.predict(X_cal)
        yhat_test = model.predict(X_test)

        mae = mean_absolute_error(y_test, yhat_test)
        r2 = r2_score(y_test, yhat_test) if len(set(y_test)) > 1 else float("nan")

        lo_s, hi_s, _ = split_conformal_intervals(y_cal, yhat_cal, y_test, yhat_test, alpha=ALPHA)
        cov_s = empirical_coverage(y_test, lo_s, hi_s)
        width_s = mean_interval_width(lo_s, hi_s)

        lo_n, hi_n, _, _ = normalized_conformal_intervals(X_cal, y_cal, yhat_cal, X_test, yhat_test, alpha=ALPHA, seed=seed)
        cov_n = empirical_coverage(y_test, lo_n, hi_n)
        width_n = mean_interval_width(lo_n, hi_n)

        rows.append({
            "regime": regime, "test_battery": test_id, "model": model_name,
            "n_test": len(y_test), "MAE_cycles": mae, "R2": r2,
            "target_coverage": 1 - ALPHA,
            "std_coverage": cov_s, "std_width": width_s,
            "adaptive_coverage": cov_n, "adaptive_width": width_n,
        })

        try:
            if model_name == "ElasticNet":
                bg = X_train[np.random.choice(len(X_train), min(100, len(X_train)), replace=False)]
                explainer = shap.LinearExplainer(model.named_steps["reg"], model.named_steps["scale"].transform(bg))
                sv = explainer.shap_values(model.named_steps["scale"].transform(X_test))
            else:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_test)
            mean_abs_shap = np.abs(sv).mean(axis=0)
            for feat, val in zip(COMMON_FEATURES, mean_abs_shap):
                shap_records.append({"regime": regime, "test_battery": test_id, "model": model_name, "feature": feat, "mean_abs_shap": val})
        except Exception as e:
            print(f"  [SHAP skipped] {regime}/{model_name}/{test_id}: {e}")


def load_data():
    """
    Try to build the dataset fresh from raw NASA/CALCE files (data/raw/...).
    If those aren't present -- e.g. a fresh clone of the repo, which does
    NOT ship the raw proprietary data files -- fall back to the committed,
    already-processed results/combined_dataset.csv so the pipeline runs
    out of the box with zero setup beyond `pip install -r requirements.txt`.
    """
    try:
        df = load_all()
        print("Built dataset fresh from raw NASA/CALCE files in data/raw/.")
        return df
    except FileNotFoundError:
        cached_path = Path(RESULTS_DIR) / "combined_dataset.csv"
        print(f"Raw data not found in data/raw/ -- using the committed, "
              f"already-processed dataset at {cached_path}.\n"
              f"(To reproduce from scratch: download the NASA PCoE and CALCE "
              f"CS2 raw files per README.md and place them under data/raw/, "
              f"then rerun.)")
        return pd.read_csv(cached_path)


def run():
    df = load_data()
    nasa_ids = sorted(df[df["dataset"] == "NASA"]["battery_id"].unique())
    calce_ids = sorted(df[df["dataset"] == "CALCE"]["battery_id"].unique())
    print("NASA:", nasa_ids, " CALCE:", calce_ids)

    rows, shap_records = [], []

    # 1. within-NASA LOBO
    for test_id in nasa_ids:
        others = [b for b in nasa_ids if b != test_id]
        X_train, y_train, X_cal, y_cal, X_test, y_test = make_split(df, others, None, test_id)
        evaluate("within_NASA", test_id, X_train, y_train, X_cal, y_cal, X_test, y_test, rows, shap_records)

    # 2. within-CALCE LOBO
    for test_id in calce_ids:
        others = [b for b in calce_ids if b != test_id]
        X_train, y_train, X_cal, y_cal, X_test, y_test = make_split(df, others, None, test_id)
        evaluate("within_CALCE", test_id, X_train, y_train, X_cal, y_cal, X_test, y_test, rows, shap_records)

    # 3a. train on ALL NASA -> test on EACH CALCE cell
    for test_id in calce_ids:
        X_train, y_train, X_cal, y_cal, X_test, y_test = make_split(df, nasa_ids, None, test_id)
        evaluate("NASA_to_CALCE", test_id, X_train, y_train, X_cal, y_cal, X_test, y_test, rows, shap_records)

    # 3b. train on ALL CALCE -> test on EACH NASA cell
    for test_id in nasa_ids:
        X_train, y_train, X_cal, y_cal, X_test, y_test = make_split(df, calce_ids, None, test_id)
        evaluate("CALCE_to_NASA", test_id, X_train, y_train, X_cal, y_cal, X_test, y_test, rows, shap_records)

    results_df = pd.DataFrame(rows)
    shap_df = pd.DataFrame(shap_records)
    results_df.to_csv(f"{RESULTS_DIR}/v2_results.csv", index=False)
    shap_df.to_csv(f"{RESULTS_DIR}/v2_shap.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Summary by regime x model ===")
    summary = results_df.groupby(["regime", "model"])[
        ["MAE_cycles", "R2", "std_coverage", "std_width", "adaptive_coverage", "adaptive_width"]
    ].mean().round(3)
    print(summary)
    summary.to_csv(f"{RESULTS_DIR}/v2_summary.csv")
    return results_df, shap_df


if __name__ == "__main__":
    run()
