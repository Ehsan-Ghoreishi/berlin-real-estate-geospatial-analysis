import pandas as pd

from src.visualize import price_color, stratified_sample


def test_price_color_endpoints():
    assert price_color(100, 100, 200) == "#00ff00"
    assert price_color(200, 100, 200) == "#ff0000"


def test_price_color_clamps_out_of_range():
    assert price_color(-1, 0, 10) == price_color(0, 0, 10)
    assert price_color(99, 0, 10) == price_color(10, 0, 10)


def test_price_color_constant_range():
    assert price_color(5, 5, 5) == "#00ff00"


def test_stratified_sample_keeps_bezirk_proportions():
    df = pd.DataFrame({"bezirk": ["A"] * 800 + ["B"] * 200, "v": range(1000)})
    out = stratified_sample(df, 100)
    counts = out["bezirk"].value_counts()
    assert counts["A"] == 80
    assert counts["B"] == 20


def test_stratified_sample_is_reproducible():
    df = pd.DataFrame({"bezirk": ["A", "B"] * 50, "v": range(100)})
    assert stratified_sample(df, 20).equals(stratified_sample(df, 20))
