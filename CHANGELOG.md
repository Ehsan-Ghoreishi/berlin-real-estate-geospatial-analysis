# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-10-22

### Added
- Histogram Gradient Boosting model, now the saved best model
  (holdout R² 0.940 vs 0.891 for Random Forest; 0.7 MB vs 14.8 MB).
- `src/tune.py` randomized hyperparameter search (`make tune`).
- Executed analysis notebook in `notebooks/` (`make notebook`).
- Dockerfile and `make docker-build` / `make docker-run`; CI builds the image.
- Type hints on all functions, checked with mypy locally and in CI.

### Changed
- Generated binaries (SQLite DB, model, GeoPackage) are no longer tracked;
  `make run` rebuilds them.
- README links the exact Kaggle dataset and its Apache 2.0 licence.
- Feature importance falls back to permutation importance for models
  without built-in importances.

### Fixed
- OpenMP thread oversubscription in gradient boosting on many-core
  machines (model stage 2 min -> 50 s).

## [0.2.0] - 2026-10-22

### Added
- `yield_analysis` stage: citywide gross rental yield over time and a
  per-Ortsteil buy-vs-rent ranking from listing-level data.
- 5-fold cross-validation and an out-of-time holdout (train 2020–2024,
  test 2025–2026) in the model stage.
- Required-column schema validation in the extract stage.
- `main.py` CLI: `--stages`, `--list`, `--verbose`, per-stage timings.
- pytest suite (21 tests), ruff config, pre-commit hooks, GitHub Actions CI,
  Makefile, MIT licence.

### Changed
- Paths, seeds and CRS constants centralised in `src/config.py`.
- `print` replaced with the `logging` module.
- `model_metrics.json` now groups results under `holdout`,
  `cross_validation` and `temporal_holdout`.

### Fixed
- SQLite connection is closed even when loading a table fails.
- Removed unused `geopy` dependency and a docstring that described the
  wrong distance method.

## [0.1.0] - 2026-08-04

### Added
- ETL pipeline on the Kaggle Berlin real-estate dataset, KD-tree
  nearest-station feature, Linear Regression vs Random Forest price model,
  policy-shock timeline, Folium and static maps.
