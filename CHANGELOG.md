# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
