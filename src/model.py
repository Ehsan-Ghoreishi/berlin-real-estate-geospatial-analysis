"""
model.py
========
Trains and evaluates regression models predicting price_eur from
structural + geospatial features. Mirrors the model-selection pattern
used in the author's fraud-detection portfolio project: start with a
simple, interpretable baseline (Linear Regression) then compare to a
Random Forest, and report RMSE / R^2 -- not just a single accuracy
number -- because this is a regression, not classification, problem.

Interview-ready justification for Random Forest over a deeper model
(e.g. gradient boosting / neural net): higher interpretability via
feature_importances_, and the training set here (~1.4k rows) is small
enough that a deep model would be prone to overfitting relative to a
bagged ensemble of shallow trees.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

NUMERIC_FEATURES = [
    "size_sqm", "rooms", "year_built",
    "dist_center_km", "dist_nearest_station_km",
]
CATEGORICAL_FEATURES = ["bezirk", "condition"]
TARGET = "price_eur"


def load_data():
    df = pd.read_csv(PROC_DIR / "listings_clean.csv")
    return df


def build_pipeline(model):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def evaluate(pipe, X_test, y_test, name):
    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print(f"[model] {name:18s} RMSE={rmse:,.0f} EUR  MAE={mae:,.0f} EUR  R2={r2:.3f}")
    return {"model": name, "rmse_eur": round(rmse, 0),
            "mae_eur": round(mae, 0), "r2": round(r2, 4)}


def run():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # proper out-of-sample split -- no leakage, fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    lin_pipe = build_pipeline(LinearRegression())
    lin_pipe.fit(X_train, y_train)
    results.append(evaluate(lin_pipe, X_test, y_test, "LinearRegression"))

    rf_pipe = build_pipeline(
        RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
    )
    rf_pipe.fit(X_train, y_train)
    results.append(evaluate(rf_pipe, X_test, y_test, "RandomForest"))

    # pick best by RMSE, save it
    best = min(results, key=lambda r: r["rmse_eur"])
    best_pipe = rf_pipe if best["model"] == "RandomForest" else lin_pipe

    model_path = MODEL_DIR / "price_model.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"[model] best model = {best['model']} -> saved to {model_path}")

    # interpretability output for the write-up / interview -- works
    # for whichever model won
    winning_pipe = rf_pipe if best["model"] == "RandomForest" else lin_pipe
    ohe = winning_pipe.named_steps["pre"].named_transformers_["cat"]
    feat_names = NUMERIC_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))

    if best["model"] == "RandomForest":
        importances = winning_pipe.named_steps["model"].feature_importances_
        top = sorted(zip(feat_names, importances), key=lambda t: -t[1])[:10]
        print("[model] top feature importances (RandomForest):")
        for f, imp in top:
            print(f"    {f:30s} {imp:.4f}")
    else:
        coefs = winning_pipe.named_steps["model"].coef_
        top = sorted(zip(feat_names, coefs), key=lambda t: -abs(t[1]))[:10]
        print("[model] top standardized coefficients (LinearRegression):")
        for f, c in top:
            print(f"    {f:30s} {c:+,.0f}")

    metrics_path = PROC_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[model] metrics -> {metrics_path}")

    return results


if __name__ == "__main__":
    run()
