import sqlite3

import pandas as pd
import pytest

from src import load
from src.extract import REQUIRED_COLUMNS, SchemaError, validate_file


def test_validate_file_rejects_missing_columns():
    df = pd.DataFrame({"station_name": ["Alex"], "lat": [52.5]})
    with pytest.raises(SchemaError, match="line"):
        validate_file("transit_stations.csv", df)


def test_validate_file_warns_on_row_count(caplog):
    df = pd.DataFrame({c: [0] for c in REQUIRED_COLUMNS["transit_stations.csv"]})
    validate_file("transit_stations.csv", df)
    assert "expected 135" in caplog.text


def test_load_writes_tables_and_indexes(tmp_path):
    for fname in load.TABLES.values():
        pd.DataFrame(
            {
                "bezirk": ["Mitte", "Pankow"],
                "ortsteil": ["Mitte (Ort)", "Prenzlauer Berg"],
                "price_per_m2_eur": [7000, 6000],
            }
        ).to_csv(tmp_path / fname, index=False)

    db = load.run(db_path=tmp_path / "test.db", proc_dir=tmp_path)

    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert tables == set(load.TABLES)
    assert {"idx_sales_bezirk", "idx_rentals_bezirk", "idx_kiez_ortsteil"} <= indexes
