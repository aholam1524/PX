"""Tests for area detail panel helpers (no Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.data.prices import parse_json_stat2
from housing_analyzer.panel import (
    area_detail_export_frame,
    build_trend_chart_data,
    flag_unusual_quarter_changes,
    index_series_to_100,
    municipality_name_from_prices,
    quarter_on_quarter_changes,
    quarterly_area_prices,
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


def test_municipality_name_from_prices_parentheses(sample_prices_frame):
    assert municipality_name_from_prices(sample_prices_frame, "00100") == "Helsinki"
    assert municipality_name_from_prices(sample_prices_frame, "01200") == "Vantaa"


def test_index_series_to_100_first_valid_and_missing():
    series = pd.Series({"2020Q1": float("nan"), "2020Q2": 200.0, "2020Q3": 250.0})
    indexed = index_series_to_100(series)
    assert np.isnan(indexed["2020Q1"])
    assert indexed["2020Q2"] == pytest.approx(100.0)
    assert indexed["2020Q3"] == pytest.approx(125.0)


def test_index_series_to_100_constant_series():
    series = pd.Series({"2020Q1": 500.0, "2020Q2": 500.0})
    indexed = index_series_to_100(series)
    assert indexed["2020Q1"] == pytest.approx(100.0)
    assert indexed["2020Q2"] == pytest.approx(100.0)


def test_index_series_to_100_zero_base():
    series = pd.Series({"2020Q1": 0.0, "2020Q2": 10.0})
    indexed = index_series_to_100(series)
    assert np.isnan(indexed["2020Q2"])


def test_quarter_on_quarter_skips_gaps():
    series = pd.Series({"2024Q3": 100.0, "2024Q4": 110.0})
    qoq = quarter_on_quarter_changes(series)
    assert np.isnan(qoq["2024Q3"])
    assert qoq["2024Q4"] == pytest.approx(10.0)


def test_flag_unusual_short_series_returns_empty():
    series = pd.Series({f"2020Q{i}": 100.0 + i for i in range(1, 5)})
    assert flag_unusual_quarter_changes(series) == []


def test_flag_unusual_constant_series_zero_std():
    labels = []
    year, q = 2020, 1
    for _ in range(13):
        labels.append(f"{year}Q{q}")
        q += 1
        if q > 4:
            q, year = 1, year + 1
    series = pd.Series({label: 1000.0 for label in labels})
    assert flag_unusual_quarter_changes(series) == []


def test_flag_unusual_detects_spike():
    labels = []
    year = 2020
    q = 1
    for i in range(14):
        labels.append(f"{year}Q{q}")
        q += 1
        if q > 4:
            q = 1
            year += 1
    values = [100.0] * 14
    values[-1] = 200.0
    series = pd.Series(dict(zip(labels, values)))
    flagged = flag_unusual_quarter_changes(series)
    assert flagged
    assert flagged[-1]["quarter"] == labels[-1]
    assert flagged[-1]["threshold_qoq"] == pytest.approx(
        3.0 * flagged[-1]["std_qoq"]
    )


def test_build_trend_chart_data_municipality_unavailable(sample_prices_frame, sample_boundaries):
    data = build_trend_chart_data(
        sample_prices_frame,
        sample_boundaries,
        "00100",
        building_type=None,
    )
    assert not data.municipality_available
    assert data.municipality_prices is None
    assert len(data.quarters) == len(data.area_prices)
    assert None in data.area_prices or any(p is not None for p in data.area_prices)


def test_build_trend_chart_data_with_municipality_mapping(sample_prices_frame):
    boundaries = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"postal_code": "00100", "kunta": "091"},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            },
            {
                "type": "Feature",
                "properties": {"postal_code": "00120", "kunta": "091"},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            },
            {
                "type": "Feature",
                "properties": {"postal_code": "01200", "kunta": "092"},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            },
        ],
    }
    data = build_trend_chart_data(
        sample_prices_frame,
        boundaries,
        "00100",
        building_type=None,
    )
    assert data.municipality_available
    assert data.municipality_prices is not None


def test_trend_chart_index_mode(sample_prices_frame, sample_boundaries):
    raw = build_trend_chart_data(
        sample_prices_frame, sample_boundaries, "00100", None, index_to_100=False
    )
    indexed = build_trend_chart_data(
        sample_prices_frame, sample_boundaries, "00100", None, index_to_100=True
    )
    assert indexed.indexed
    first_area = next(v for v in indexed.area_prices if v is not None)
    assert first_area == pytest.approx(100.0)
    assert raw.indexed is False


def test_area_export_frame_filters_building_type(sample_prices_frame):
    all_rows = area_detail_export_frame(sample_prices_frame, "00100", None)
    bt = sample_prices_frame.loc[
        sample_prices_frame["postal_code"] == "00100", "building_type"
    ].iloc[0]
    one_type = area_detail_export_frame(sample_prices_frame, "00100", bt)
    assert len(one_type) <= len(all_rows)
    if not one_type.empty:
        assert one_type["building_type"].nunique() == 1


def test_quarterly_area_prices_missing_not_zero(sample_prices_frame):
    series = quarterly_area_prices(sample_prices_frame, "99999", None)
    assert series.isna().all()
