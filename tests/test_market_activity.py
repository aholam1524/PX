"""Tests for market activity (sales per 1,000 inhabitants)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.analysis.market_activity import (
    MARKET_ACTIVITY_LOW_RELIABILITY_CAPTION,
    market_activity,
    market_activity_reliability_note,
    market_activity_table,
)
from housing_analyzer.data.prices import parse_json_stat2
from housing_analyzer.map import (
    METRIC_MARKET_ACTIVITY,
    format_hover_text,
    metric_is_missing,
    prepare_map_dataframe,
    trailing_sales_by_area,
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


@pytest.fixture
def demographics_sample() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "demographics_sample.csv")


@pytest.fixture
def cpi_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-10-01", "2024-11-01", "2024-12-01"]),
            "cpi": [102.0, 102.2, 102.4],
        }
    )


def test_market_activity_hand_calculated_ratio():
    # 25 sales / 5000 people * 1000 = 5.0
    assert market_activity(25.0, 5000.0) == pytest.approx(5.0)


def test_market_activity_missing_sales():
    assert np.isnan(market_activity(float("nan"), 5000.0))


def test_market_activity_missing_population():
    assert np.isnan(market_activity(10.0, float("nan")))


def test_market_activity_population_below_minimum():
    assert np.isnan(market_activity(10.0, 50.0, min_population=100))


def test_market_activity_zero_sales_is_valid():
    assert market_activity(0.0, 5000.0) == pytest.approx(0.0)


def test_market_activity_reliability_note_under_threshold():
    assert market_activity_reliability_note(9.0) == MARKET_ACTIVITY_LOW_RELIABILITY_CAPTION
    assert market_activity_reliability_note(10.0) is None
    assert market_activity_reliability_note(float("nan")) is None


def test_market_activity_table_joins_sales_and_population(
    sample_prices_frame, demographics_sample
):
    sales = trailing_sales_by_area(sample_prices_frame, "2024Q4", "1")
    table = market_activity_table(sales, demographics_sample)
    assert set(table.columns) >= {"postal_code", "sales_4q", "population", "market_activity"}
    row = table.loc[table["postal_code"] == "00100"].iloc[0]
    expected = market_activity(row["sales_4q"], row["population"])
    assert row["market_activity"] == pytest.approx(expected)


def test_prepare_map_dataframe_market_activity_layer(
    sample_prices_frame, sample_boundaries, demographics_sample, cpi_df
):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_MARKET_ACTIVITY,
        cpi_df=cpi_df,
        demographics_df=demographics_sample,
    )
    row = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    assert row["market_activity"] == pytest.approx(
        market_activity(row["sales_4q"], row["population"])
    )
    assert not bool(row["missing"])
    hover = format_hover_text(row, METRIC_MARKET_ACTIVITY)
    assert "sales in the last four quarters" in hover
    assert "inhabitants" in hover


def test_market_activity_missing_not_zero(
    sample_prices_frame, sample_boundaries, demographics_sample, cpi_df
):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_MARKET_ACTIVITY,
        cpi_df=cpi_df,
        demographics_df=demographics_sample,
    )
    row = frame.loc[frame["postal_code"] == "01200"].iloc[0]
    assert metric_is_missing(row, METRIC_MARKET_ACTIVITY)
    assert np.isnan(row["market_activity"])
