"""Tests for compare tab helpers and similar-area search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.analysis.compare import build_comparison_table
from housing_analyzer.analysis.similar_areas import SIMILARITY_FEATURES, similar_areas


def _summaries_frame(rows: dict[str, dict]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "postal_code"
    return frame


def test_similar_areas_known_distances_and_excludes_target():
    summaries = _summaries_frame(
        {
            "00100": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 2.0,
                "pct_change_5y": 10.0,
                "reliability": "ok",
                "rank": 1,
                "percentile": 100.0,
            },
            "00200": {
                "price_per_sqm": 5100.0,
                "pct_change_1y": 2.0,
                "pct_change_5y": 10.5,
                "reliability": "ok",
                "rank": 2,
                "percentile": 90.0,
            },
            "00300": {
                "price_per_sqm": 7000.0,
                "pct_change_1y": 2.0,
                "pct_change_5y": 30.0,
                "reliability": "ok",
                "rank": 3,
                "percentile": 80.0,
            },
        }
    )
    out = similar_areas(summaries, "00100", limit=5)
    codes = [item.postal_code for item in out]
    assert "00100" not in codes
    assert codes == ["00200", "00300"]
    assert out[0].distance < out[1].distance


def test_similar_areas_missing_feature_drops_from_pool():
    summaries = _summaries_frame(
        {
            "00100": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 1.0,
                "pct_change_5y": 10.0,
                "reliability": "ok",
                "rank": 1,
                "percentile": 100.0,
            },
            "00200": {
                "price_per_sqm": 5100.0,
                "pct_change_1y": 1.0,
                "pct_change_5y": float("nan"),
                "reliability": "ok",
                "rank": 2,
                "percentile": 90.0,
            },
            "00300": {
                "price_per_sqm": 5200.0,
                "pct_change_1y": 1.0,
                "pct_change_5y": 11.0,
                "reliability": "ok",
                "rank": 3,
                "percentile": 80.0,
            },
        }
    )
    out = similar_areas(summaries, "00100")
    assert [item.postal_code for item in out] == ["00300"]


def test_similar_areas_tie_breaks_on_postal_code():
    summaries = _summaries_frame(
        {
            "00100": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 0.0,
                "pct_change_5y": 10.0,
                "reliability": "ok",
                "rank": 1,
                "percentile": 100.0,
            },
            "00200": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 0.0,
                "pct_change_5y": 10.0,
                "reliability": "ok",
                "rank": 2,
                "percentile": 90.0,
            },
            "00300": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 0.0,
                "pct_change_5y": 10.0,
                "reliability": "ok",
                "rank": 3,
                "percentile": 80.0,
            },
        }
    )
    out = similar_areas(summaries, "00100", limit=5)
    assert [item.postal_code for item in out] == ["00200", "00300"]
    assert out[0].distance == pytest.approx(out[1].distance)


def test_similar_areas_fewer_than_five():
    summaries = _summaries_frame(
        {
            "00100": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 0.0,
                "pct_change_5y": 10.0,
                "reliability": "ok",
                "rank": 1,
                "percentile": 100.0,
            },
            "00200": {
                "price_per_sqm": 5200.0,
                "pct_change_1y": 0.0,
                "pct_change_5y": 12.0,
                "reliability": "ok",
                "rank": 2,
                "percentile": 90.0,
            },
        }
    )
    assert len(similar_areas(summaries, "00100", limit=5)) == 1


def test_similar_areas_low_reliability_not_in_pool():
    summaries = _summaries_frame(
        {
            "00100": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 0.0,
                "pct_change_5y": 10.0,
                "reliability": "ok",
                "rank": 1,
                "percentile": 100.0,
            },
            "00200": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 0.0,
                "pct_change_5y": 10.0,
                "reliability": "low",
                "rank": 2,
                "percentile": 90.0,
            },
        }
    )
    assert similar_areas(summaries, "00100") == []


def test_build_comparison_table_columns_per_area():
    summaries = _summaries_frame(
        {
            "00100": {
                "price_per_sqm": 5000.0,
                "pct_change_1y": 2.0,
                "pct_change_5y": 10.0,
                "pct_change_1y_real": 1.0,
                "pct_change_5y_real": 8.0,
                "reliability": "ok",
                "rank": 3,
                "percentile": 75.0,
            },
            "00200": {
                "price_per_sqm": 4000.0,
                "pct_change_1y": -1.0,
                "pct_change_5y": 5.0,
                "pct_change_1y_real": -2.0,
                "pct_change_5y_real": 3.0,
                "reliability": "low",
                "rank": 10,
                "percentile": 40.0,
            },
        }
    )
    summaries["pct_change_1y_real"] = summaries.get("pct_change_1y_real", float("nan"))
    sales = pd.Series({"00100": 42.0, "00200": 8.0})
    table = build_comparison_table(
        summaries, sales, ["00100", "00200"], include_real=True
    )
    assert list(table.columns) == ["00100", "00200"]
    assert table.loc["Price per m²", "00100"] == "5,000 EUR/m²"
    assert table.loc["1-year change (nominal)", "00200"] == "-1.0%"
    assert table.loc["Sales (last 4 quarters)", "00100"] == "42"
    assert table.loc["Reliability", "00200"] == "low"
    assert table.loc["Rank", "00100"] == "3"


def test_similarity_features_include_price_and_five_year_change():
    names = {name for name, _ in SIMILARITY_FEATURES}
    assert "price_per_sqm" in names
    assert "pct_change_5y" in names
