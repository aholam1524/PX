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
    postal_code_from_selection,
    prepare_map_dataframe,
    search_area_matches,
    trailing_sales_by_area,
    trailing_sales_sum,
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


def test_prepare_map_dataframe_does_not_compute_per_area_summaries(
    sample_prices_frame, sample_boundaries, monkeypatch
):
    """The map must use the all-areas code path, not one full ranking per area."""
    import housing_analyzer.analysis.metrics as metrics

    def boom(*args, **kwargs):
        raise AssertionError("per-area summary used while preparing the map")

    monkeypatch.setattr(metrics, "summarize_area", boom)
    monkeypatch.setattr(metrics, "rank_percentile", boom)
    frame = prepare_map_dataframe(
        sample_prices_frame, sample_boundaries, "2024Q4", "all", METRIC_PRICE
    )
    assert not frame.empty


def test_prepare_map_dataframe_matches_per_area_values(
    sample_prices_frame, sample_boundaries
):
    from housing_analyzer.analysis.metrics import summarize_area
    from housing_analyzer.map import resolve_building_type_label

    for code in ("all", "1", "2"):
        frame = prepare_map_dataframe(
            sample_prices_frame, sample_boundaries, "2024Q4", code, METRIC_PRICE
        )
        label = resolve_building_type_label(
            sample_prices_frame, None if code == "all" else code
        )
        for _, row in frame.iterrows():
            single = summarize_area(
                sample_prices_frame, row["postal_code"], "2024Q4", building_type=label
            )
            if np.isnan(single["price_per_sqm"]):
                assert np.isnan(row["price_per_sqm"])
            else:
                assert row["price_per_sqm"] == pytest.approx(single["price_per_sqm"])
            assert (row["reliability"] or None) == single["reliability"]


def test_trailing_sales_by_area_matches_single_area_sum(sample_prices_frame):
    totals = trailing_sales_by_area(sample_prices_frame, "2024Q4", "1")
    for postal, value in totals.items():
        assert trailing_sales_sum(
            sample_prices_frame, postal, "2024Q4", "1"
        ) == pytest.approx(value)
    assert np.isnan(trailing_sales_sum(sample_prices_frame, "99999", "2024Q4", "1"))


def test_search_area_matches_by_code_and_name_case_insensitive():
    frame = pd.DataFrame(
        {
            "postal_code": ["00100", "00120", "02100"],
            "area_name": ["Helsinki keskusta", "Punavuori", "Tapiola"],
        }
    )
    assert search_area_matches(frame, "0010") == ["00100"]
    assert search_area_matches(frame, "punavuori") == ["00120"]
    assert search_area_matches(frame, "  HELSINKI ") == ["00100"]
    assert search_area_matches(frame, "001") == ["00100", "00120"]
    assert search_area_matches(frame, "zzz") == []
    assert search_area_matches(frame, "") == []
    assert search_area_matches(frame.iloc[0:0], "001") == []


def test_search_area_matches_treats_special_characters_literally():
    frame = pd.DataFrame(
        {"postal_code": ["00100", "00120"], "area_name": ["A (centre)", "B+C"]}
    )
    assert search_area_matches(frame, "(centre)") == ["00100"]
    assert search_area_matches(frame, "b+c") == ["00120"]


class _Event:
    """Stands in for the selection event object that Streamlit returns."""

    def __init__(self, selection):
        self.selection = selection


def test_postal_code_from_selection_prefers_customdata_then_location():
    event = _Event({"points": [{"customdata": ["100", "Helsinki"]}]})
    assert postal_code_from_selection(event) == "00100"
    event = _Event({"points": [{"location": "2100"}]})
    assert postal_code_from_selection(event) == "02100"
    event = _Event({"points": [{"other": 1}, {"location": "00120"}]})
    assert postal_code_from_selection(event) == "00120"


def test_postal_code_from_selection_handles_empty_and_odd_input():
    assert postal_code_from_selection(None) is None
    assert postal_code_from_selection(_Event({})) is None
    assert postal_code_from_selection(_Event({"points": []})) is None
    assert (
        postal_code_from_selection({"selection": {"points": [{"location": "00100"}]}})
        == "00100"
    )
    assert postal_code_from_selection(object()) is None


def test_each_map_trace_carries_only_its_own_areas(
    sample_prices_frame, sample_boundaries
):
    frame = prepare_map_dataframe(
        sample_prices_frame, sample_boundaries, "2024Q4", "all", METRIC_PRICE
    )
    fig = build_choropleth_figure(frame, sample_boundaries, METRIC_PRICE)
    for trace in fig.data:
        codes = {str(code).zfill(5) for code in trace.locations}
        feature_codes = {
            str(feature["properties"]["postal_code"]).zfill(5)
            for feature in trace.geojson["features"]
        }
        assert feature_codes == codes
