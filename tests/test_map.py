"""Tests for map layer preparation and choropleth helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.data.prices import parse_json_stat2
from housing_analyzer.map import (
    METRIC_CHANGE_1Y,
    METRIC_PRICE,
    METRIC_SALES,
    build_choropleth_figure,
    format_hover_text,
    latest_quarter_with_data,
    metric_color_range,
    metric_is_missing,
    prepare_map_dataframe,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_prices_frame() -> pd.DataFrame:
    with (FIXTURES / "prices_sample.json").open(encoding="utf-8") as handle:
        return parse_json_stat2(json.load(handle))


@pytest.fixture
def sample_boundaries() -> dict:
    with (FIXTURES / "boundaries_sample.geojson").open(encoding="utf-8") as handle:
        return json.load(handle)


def test_latest_quarter_with_data(sample_prices_frame):
    assert latest_quarter_with_data(sample_prices_frame) == "2024Q4"
    assert latest_quarter_with_data(sample_prices_frame, "1") == "2024Q4"


def test_metric_color_range_pct_change_symmetric():
    values = pd.Series([-10.0, 5.0, 20.0])
    low, high = metric_color_range(values, METRIC_CHANGE_1Y)
    assert low == pytest.approx(-20.0)
    assert high == pytest.approx(20.0)


def test_missing_metric_not_treated_as_zero():
    row = {
        "postal_code": "00100",
        "area_name": "Test",
        "price_per_sqm": float("nan"),
        "pct_change_1y": float("nan"),
        "pct_change_5y": float("nan"),
        "sales_4q": float("nan"),
        "reliability": "none",
    }
    assert metric_is_missing(row, METRIC_PRICE)
    hover = format_hover_text(row, METRIC_PRICE)
    assert "No data" in hover
    assert "0 EUR" not in hover


def test_hover_includes_low_reliability_note(sample_prices_frame, sample_boundaries):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "2",
        METRIC_PRICE,
    )
    low_rows = frame.loc[frame["reliability"] == "low"]
    if not low_rows.empty:
        hover = low_rows.iloc[0]["hover"]
        assert "Based on few sales" in hover


def test_prepare_map_dataframe_marks_missing_price_grey_candidates(
    sample_prices_frame, sample_boundaries
):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_PRICE,
    )
    missing = frame.loc[frame["postal_code"] == "01200"]
    assert len(missing) == 1
    assert bool(missing.iloc[0]["missing"])


def test_sales_metric_uses_trailing_sum(sample_prices_frame, sample_boundaries):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_SALES,
    )
    row = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    assert row[METRIC_SALES] == pytest.approx(50.0)


def test_build_choropleth_figure_returns_figure(sample_prices_frame, sample_boundaries):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "all",
        METRIC_PRICE,
    )
    fig = build_choropleth_figure(frame, sample_boundaries, METRIC_PRICE)
    assert fig.data
    trace_names = {trace.name for trace in fig.data}
    assert "No data" in trace_names or "Areas with data" in trace_names


def test_color_range_ignores_nan(sample_prices_frame, sample_boundaries):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "all",
        METRIC_PRICE,
    )
    values = frame.loc[~frame["missing"], METRIC_PRICE]
    low, high = metric_color_range(values, METRIC_PRICE)
    assert not np.isnan(low)
    assert not np.isnan(high)
    assert low <= high
