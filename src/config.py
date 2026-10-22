"""
config.py
=========
Single source of truth for filesystem paths and project-wide constants,
so every pipeline stage resolves data locations the same way.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"
MAP_DIR = ROOT_DIR / "maps"

DB_PATH = PROC_DIR / "berlin_real_estate.db"
MODEL_PATH = PROC_DIR / "price_model.joblib"
METRICS_PATH = PROC_DIR / "model_metrics.json"

# Reproducibility: one seed shared by train/test splits, models and sampling
RANDOM_STATE = 42

# ETRS89 / UTM zone 33N -- metric CRS appropriate for Berlin distances
METRIC_CRS = 25833
WGS84_CRS = 4326

BRANDENBURG_GATE = (52.5163, 13.3777)


def ensure_dirs() -> None:
    """Create output directories if they don't exist yet."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
