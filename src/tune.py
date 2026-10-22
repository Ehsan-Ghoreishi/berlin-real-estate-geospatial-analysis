"""
tune.py
=======
Randomized hyperparameter search for the gradient-boosting price model.

Kept out of the default pipeline because it fits ~45 models: run it on
demand (`make tune` / `python3 -m src.tune`) and copy the winning
parameters into `candidate_models()` in model.py if they beat the
current defaults by more than the CV noise.

Search uses 3-fold CV on the training split only; the 20% holdout is
scored once at the end so the reported number isn't tuned on.
"""

import json
import logging

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from threadpoolctl import threadpool_limits

from src.config import PROC_DIR, RANDOM_STATE
from src.model import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OPENMP_THREADS,
    TARGET,
    build_pipeline,
    candidate_models,
    evaluate,
    load_data,
)

logger = logging.getLogger(__name__)

N_ITER = 15
CV_FOLDS = 3
RESULTS_PATH = PROC_DIR / "tuning_results.json"

PARAM_DISTRIBUTIONS = {
    "model__learning_rate": loguniform(0.02, 0.3),
    "model__max_leaf_nodes": randint(16, 128),
    "model__min_samples_leaf": randint(5, 80),
    "model__l2_regularization": loguniform(1e-3, 10),
    "model__max_iter": randint(200, 900),
    "model__max_features": uniform(0.5, 0.5),
}


def run(n_iter: int = N_ITER) -> dict:
    df = load_data()
    X = df[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        build_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        verbose=0,
    )
    with threadpool_limits(limits=OPENMP_THREADS, user_api="openmp"):
        search.fit(X_train, y_train)
        baseline = build_pipeline(candidate_models()["HistGradientBoosting"])
        baseline.fit(X_train, y_train)

    best_params = {
        k.removeprefix("model__"): (round(float(v), 4) if isinstance(v, float) else int(v))
        for k, v in search.best_params_.items()
    }
    logger.info(f"best CV RMSE: {-search.best_score_:,.0f} EUR")
    logger.info(f"best params: {best_params}")

    tuned = evaluate(search.best_estimator_, X_test, y_test, "HGB tuned")
    default = evaluate(baseline, X_test, y_test, "HGB default")

    results = {
        "n_iter": n_iter,
        "cv_folds": CV_FOLDS,
        "best_cv_rmse_eur": round(-search.best_score_, 0),
        "best_params": best_params,
        "holdout_tuned": tuned,
        "holdout_default": default,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"wrote -> {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
