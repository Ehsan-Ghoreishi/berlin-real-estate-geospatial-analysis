"""
model.py
========
Trains Linear Regression and Random Forest models to predict secondary-
market sale price (price_eur) from structural, building-quality, macro,
and geospatial features -- a much richer feature set than the original
synthetic prototype, since this data carries real energy-class,
building-era, condition, and mortgage-rate-at-listing information.

Two models are compared and the better one (by RMSE) is kept, per the
same rationale as before: Linear Regression for a transparent baseline,
Random Forest for capturing nonlinear interactions (e.g. energy class
mattering more after the 2024 Heizungsgesetz, or transit proximity
mattering more in outer Bezirke than central ones) without the
overfitting risk of a deeper model on this feature set.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def load_data():
    df = pd.read_csv(PROC_DIR / "sales_clean.csv")
    for c in BOOLEAN_FEATURES:
        df[c] = df[c].astype(int)
    return df


def build_pipeline(model):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + BOOLEAN_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def candidate_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def cross_validate_model(model, X, y, name, folds=CV_FOLDS):
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
        f"{name:18s} {folds}-fold CV  RMSE={rmse.mean():>10,.0f} "
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


def temporal_holdout(model, df, name, cutoff=TEMPORAL_CUTOFF):
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


def evaluate(pipe, X_test, y_test, name):
    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    logger.info(f"{name:18s} RMSE={rmse:>10,.0f} EUR  MAE={mae:>10,.0f} EUR  R2={r2:.3f}")
    return {"model": name, "rmse_eur": round(rmse, 0), "mae_eur": round(mae, 0), "r2": round(r2, 4)}


def run():
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

    ohe = best_pipe.named_steps["pre"].named_transformers_["cat"]
    feat_names = (
        NUMERIC_FEATURES + BOOLEAN_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    )

    if best["model"] == "RandomForest":
        importances = best_pipe.named_steps["model"].feature_importances_
        top = sorted(zip(feat_names, importances), key=lambda t: -t[1])[:12]
        logger.info("top feature importances (RandomForest):")
        for f, imp in top:
            logger.info(f"    {f:40s} {imp:.4f}")
    else:
        coefs = best_pipe.named_steps["model"].coef_
        top = sorted(zip(feat_names, coefs), key=lambda t: -abs(t[1]))[:12]
        logger.info("top standardized coefficients (LinearRegression):")
        for f, c in top:
            logger.info(f"    {f:40s} {c:+,.0f}")

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
