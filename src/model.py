"""
model.py
========
Trains Linear Regression, Random Forest and Histogram Gradient Boosting
models to predict secondary-market sale price (price_eur) from structural,
building-quality, macro, and geospatial features -- a much richer feature set than the original
synthetic prototype, since this data carries real energy-class,
building-era, condition, and mortgage-rate-at-listing information.

The models are compared and the best one (by holdout RMSE) is kept, per the
same rationale as before: Linear Regression for a transparent baseline,
Random Forest for capturing nonlinear interactions (e.g. energy class
mattering more after the 2024 Heizungsgesetz, or transit proximity
mattering more in outer Bezirke than central ones) without the
overfitting risk of a deeper model on this feature set, and gradient
boosting, which fits those interactions sequentially and is both more
accurate and ~20x smaller on disk than the forest here.
"""

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.config import METRICS_PATH, MODEL_PATH, PROC_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "rooms",
    "area_m2",
    "floor",
    "total_floors",
    "year_built",
    "computed_dist_nearest_station_km",
    "to_brandenburg_gate_km",
    "mortgage_rate_at_listing",
    "months_since_2020",
]
CATEGORICAL_FEATURES = [
    "bezirk",
    "building_era",
    "energy_class",
    "position",
    "condition",
    "property_type",
]
BOOLEAN_FEATURES = ["has_lift", "has_balcony", "has_cellar", "has_parking"]
TARGET = "price_eur"
CV_FOLDS = 5
TEMPORAL_CUTOFF = "2025-01-01"
# HistGradientBoosting's OpenMP threads contend badly on many-core machines
# at this data size (a fit took 13s with 14 threads vs 4s with 4)
OPENMP_THREADS = 4


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROC_DIR / "sales_clean.csv")
    for c in BOOLEAN_FEATURES:
        df[c] = df[c].astype(int)
    return df


def build_pipeline(model: BaseEstimator) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + BOOLEAN_FEATURES),
            # dense output: HistGradientBoosting does not accept sparse input
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def candidate_models() -> dict[str, BaseEstimator]:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.08,
            max_leaf_nodes=48,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def cross_validate_model(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, name: str, folds: int = CV_FOLDS
) -> dict[str, Any]:
    """K-fold CV on the full dataset: a single 80/20 split can be lucky,
    so report mean +/- std across folds to show the score is stable."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        build_pipeline(model),
        X,
        y,
        cv=cv,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
    )
    rmse = -scores["test_rmse"]
    r2 = scores["test_r2"]
    logger.info(
        f"{name:20s} {folds}-fold CV  RMSE={rmse.mean():>10,.0f} "
        f"+/- {rmse.std():,.0f} EUR  R2={r2.mean():.3f} +/- {r2.std():.3f}"
    )
    return {
        "model": name,
        "folds": folds,
        "rmse_eur_mean": round(float(rmse.mean()), 0),
        "rmse_eur_std": round(float(rmse.std()), 0),
        "r2_mean": round(float(r2.mean()), 4),
        "r2_std": round(float(r2.std()), 4),
    }


def temporal_holdout(
    model: BaseEstimator, df: pd.DataFrame, name: str, cutoff: str = TEMPORAL_CUTOFF
) -> dict[str, Any]:
    """Out-of-time check: train on listings before `cutoff`, test on
    listings after it. Random splits let the model see neighbouring
    months of the same market regime; this asks the harder question of
    how well it extrapolates into a period it has never seen."""
    dates = pd.to_datetime(df["date_listed"])
    train, test = df[dates < cutoff], df[dates >= cutoff]
    features = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    pipe = build_pipeline(model)
    pipe.fit(train[features], train[TARGET])
    result = evaluate(pipe, test[features], test[TARGET], f"{name} (t>={cutoff[:4]})")
    result.update(model=name, cutoff=cutoff, n_train=len(train), n_test=len(test))
    return result


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict[str, Any]:
    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    logger.info(f"{name:20s} RMSE={rmse:>10,.0f} EUR  MAE={mae:>10,.0f} EUR  R2={r2:.3f}")
    return {"model": name, "rmse_eur": round(rmse, 0), "mae_eur": round(mae, 0), "r2": round(r2, 4)}


def log_top_features(
    pipe: Pipeline, name: str, X_test: pd.DataFrame, y_test: pd.Series, top_n: int = 12
) -> None:
    """Tree ensembles with impurity importances and linear models with
    coefficients report those directly; anything else (e.g. gradient
    boosting) falls back to permutation importance on the raw columns."""
    model = pipe.named_steps["model"]
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    feat_names = (
        NUMERIC_FEATURES + BOOLEAN_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    )
    if hasattr(model, "feature_importances_"):
        pairs, label, fmt = (
            zip(feat_names, model.feature_importances_, strict=True),
            "importances",
            ".4f",
        )
    elif hasattr(model, "coef_"):
        pairs, label, fmt = (
            zip(feat_names, model.coef_, strict=True),
            "standardized coefficients",
            "+,.0f",
        )
    else:
        # a 5k-row sample keeps this to a few seconds; ranking is stable at that size
        sample = X_test.sample(min(len(X_test), 5000), random_state=RANDOM_STATE)
        perm = permutation_importance(
            pipe, sample, y_test.loc[sample.index], n_repeats=5, random_state=RANDOM_STATE
        )
        pairs = zip(X_test.columns, perm.importances_mean, strict=True)
        label, fmt = "permutation importances (R2 drop)", ".4f"
    top = sorted(pairs, key=lambda t: -abs(t[1]))[:top_n]
    logger.info(f"top feature {label} ({name}):")
    for f, v in top:
        logger.info(f"    {f:40s} {v:{fmt}}")


def run() -> dict[str, list[dict[str, Any]]]:
    with threadpool_limits(limits=OPENMP_THREADS, user_api="openmp"):
        return _run()


def _run() -> dict[str, list[dict[str, Any]]]:
    df = load_data()
    X = df[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results, cv_results, temporal_results, fitted = [], [], [], {}
    for name, estimator in candidate_models().items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append(evaluate(pipe, X_test, y_test, name))
        cv_results.append(cross_validate_model(estimator, X, y, name))
        temporal_results.append(temporal_holdout(estimator, df, name))

    best = min(results, key=lambda r: r["rmse_eur"])
    best_pipe = fitted[best["model"]]
    joblib.dump(best_pipe, MODEL_PATH, compress=3)
    model_size_mb = MODEL_PATH.stat().st_size / 1e6
    logger.info(f"best model = {best['model']} -> saved ({model_size_mb:.1f} MB)")

    log_top_features(best_pipe, best["model"], X_test, y_test)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "holdout": results,
                "cross_validation": cv_results,
                "temporal_holdout": temporal_results,
            },
            f,
            indent=2,
        )
    logger.info(f"metrics -> {METRICS_PATH}")
    return {
        "holdout": results,
        "cross_validation": cv_results,
        "temporal_holdout": temporal_results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
