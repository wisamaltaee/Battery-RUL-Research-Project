"""
models.py — Model zoo for RUL point prediction.

Three model families spanning the bias/variance and interpretability
spectrum, matching what the RUL literature typically benchmarks:
  - ElasticNet:      linear baseline, fully transparent
  - RandomForest:    bagged trees, moderate flexibility
  - XGBoost:         boosted trees, typically strongest point-accuracy

Comparing conformal calibration + SHAP across these three (rather than
just picking the "best" model and reporting SHAP on it alone) is the
methodological piece most single-model RUL papers skip.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb


def build_models(seed: int = 0):
    models = {
        "ElasticNet": Pipeline([
            ("scale", StandardScaler()),
            ("reg", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=seed, max_iter=5000)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            random_state=seed, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=-1,
        ),
    }
    return models
