"""Tests for map layer preparation and choropleth helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.data.prices import parse_json_stat2
from housing_analyzer.map import (
    LOW_RELIABILITY_OUTLINE_COLOR,
    MAP_TOP_MARGIN,
    METRIC_CHANGE_1Y,
    METRIC_CHANGE_1Y_REAL,
    METRIC_CHANGE_5Y,
    METRIC_CHANGE_5Y_REAL,
    METRIC_PRICE,
    METRIC_PRICE_TO_INCOME,
    METRIC_SALES,
    NO_CPI_HOVER,
    NO_DATA_HOVER,
    NO_DATA_FILL,
    VALUE_COLORSCALE,
    _hex_luminance,
    build_choropleth_figure,
    plotly_map_chart_config,
    value_colorbar,
    default_map_quarter,
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


@pytest.fixture
def cpi_df() -> pd.DataFrame:
    """CPI covering only 2024Q4 (the quarter the price fixtures use), fully complete."""
    return pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-10-01", "2024-11-01", "2024-12-01"]),
            "cpi": [102.0, 102.2, 102.4],
        }
    )


def test_latest_quarter_with_data(sample_prices_frame):
    assert latest_quarter_with_data(sample_prices_frame) == "2024Q4"
    assert latest_quarter_with_data(sample_prices_frame, "1") == "2024Q4"


def test_metric_color_range_pct_change_symmetric():
    values = pd.Series([-10.0, 5.0, 20.0])
    low, high = metric_color_range(values, METRIC_CHANGE_1Y, use_full_range=True)
    assert low == pytest.approx(-20.0)
    assert high == pytest.approx(20.0)


def test_metric_color_range_clips_outlier_narrower_than_min_max():
    values = pd.Series([1000.0, 1100.0, 1200.0, 1300.0, 50_000.0])
    low_clip, high_clip = metric_color_range(values, METRIC_PRICE)
    low_full, high_full = metric_color_range(values, METRIC_PRICE, use_full_range=True)
    assert low_full == pytest.approx(1000.0)
    assert high_full == pytest.approx(50_000.0)
    assert high_clip - low_clip < high_full - low_full


def test_metric_color_range_percentile_zero_and_hundred_equals_min_max():
    values = pd.Series([10.0, 20.0, 30.0, 40.0])
    low, high = metric_color_range(
        values, METRIC_PRICE, percentile_low=0, percentile_high=100
    )
    assert low == pytest.approx(10.0)
    assert high == pytest.approx(40.0)


def test_metric_color_range_change_layers_symmetric_with_clip():
    values = pd.Series([-30.0, -5.0, 8.0, 25.0, 100.0])
    low, high = metric_color_range(values, METRIC_CHANGE_5Y)
    assert low == pytest.approx(-high)
    assert high >= 1.0


def test_metric_color_range_empty_and_all_equal():
    low, high = metric_color_range(pd.Series(dtype=float), METRIC_PRICE)
    assert low == 0.0
    assert high == 1.0
    low, high = metric_color_range(pd.Series([5.0, 5.0, 5.0]), METRIC_PRICE)
    assert low == pytest.approx(5.0)
    assert high == pytest.approx(6.0)


def _synthetic_prices_quarters(
    quarter_counts: list[tuple[str, int]], *, base_code: int = 10000
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for quarter, n_areas in quarter_counts:
        for i in range(n_areas):
            rows.append(
                {
                    "postal_code": f"{base_code + i:05d}",
                    "quarter": quarter,
                    "building_type": "1 — test",
                    "price_per_sqm": 2000.0,
                    "transactions": 10,
                }
            )
    return pd.DataFrame(rows)


def test_default_map_quarter_skips_thin_newest_quarter():
    # Prior quarters ~100 areas; newest only ~80.
    history = [(f"2024Q{i}", 100) for i in range(1, 5)]
    history.append(("2025Q1", 80))
    df = _synthetic_prices_quarters(history)
    assert default_map_quarter(df) == "2024Q4"


def test_default_map_quarter_picks_newest_when_coverage_similar():
    history = [(f"2024Q{i}", 100) for i in range(1, 4)] + [("2024Q4", 95), ("2025Q1", 98)]
    df = _synthetic_prices_quarters(history)
    assert default_map_quarter(df) == "2025Q1"


def test_default_map_quarter_short_history_falls_back_to_latest():
    df = _synthetic_prices_quarters([("2024Q4", 50)])
    assert default_map_quarter(df) == "2024Q4"


def test_default_map_quarter_empty_returns_none():
    assert default_map_quarter(pd.DataFrame()) is None


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


def test_hover_includes_low_reliability_note(
    sample_prices_frame, sample_boundaries, cpi_df
):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "2",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    low_rows = frame.loc[frame["reliability"] == "low"]
    if not low_rows.empty:
        hover = low_rows.iloc[0]["hover"]
        assert "Based on few sales" in hover


def test_prepare_map_dataframe_marks_missing_price_grey_candidates(
    sample_prices_frame, sample_boundaries, cpi_df
):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    missing = frame.loc[frame["postal_code"] == "01200"]
    assert len(missing) == 1
    assert bool(missing.iloc[0]["missing"])


def test_sales_metric_uses_trailing_sum(sample_prices_frame, sample_boundaries, cpi_df):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_SALES,
        cpi_df=cpi_df,
    )
    row = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    assert row[METRIC_SALES] == pytest.approx(50.0)


def _value_layer_traces(fig):
    return [t for t in fig.data if t.name == "Areas with data"]


def _no_data_traces(fig):
    return [t for t in fig.data if t.name == "No data"]


def _normalize_colorscale(scale) -> list[list[Any]]:
    return [[float(stop[0]), str(stop[1])] for stop in scale]


def test_value_layers_use_light_to_dark_greyscale_not_viridis(
    sample_prices_frame, sample_boundaries, cpi_df
):
    metrics = (
        METRIC_PRICE,
        METRIC_CHANGE_1Y,
        METRIC_CHANGE_5Y,
        METRIC_CHANGE_1Y_REAL,
        METRIC_CHANGE_5Y_REAL,
        METRIC_SALES,
        METRIC_PRICE_TO_INCOME,
    )
    low_lum = _hex_luminance(VALUE_COLORSCALE[0][1])
    high_lum = _hex_luminance(VALUE_COLORSCALE[1][1])
    assert low_lum > high_lum
    for metric in metrics:
        frame = prepare_map_dataframe(
            sample_prices_frame,
            sample_boundaries,
            "2024Q4",
            "all",
            metric,
            cpi_df=cpi_df,
        )
        fig = build_choropleth_figure(frame, sample_boundaries, metric)
        for trace in _value_layer_traces(fig):
            assert trace.colorscale != "Viridis"
            assert _normalize_colorscale(trace.colorscale) == VALUE_COLORSCALE
            stops = _normalize_colorscale(trace.colorscale)
            assert _hex_luminance(stops[0][1]) > _hex_luminance(stops[-1][1])


def test_no_data_trace_is_transparent_not_from_value_scale(
    sample_prices_frame, sample_boundaries, cpi_df
):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    fig = build_choropleth_figure(frame, sample_boundaries, METRIC_PRICE)
    scale_colors = {stop[1] for stop in VALUE_COLORSCALE}
    for trace in _no_data_traces(fig):
        fill = trace.colorscale[0][1]
        assert fill == NO_DATA_FILL
        assert fill not in scale_colors


def test_low_reliability_outline_differs_from_value_scale_ends():
    scale_ends = {VALUE_COLORSCALE[0][1], VALUE_COLORSCALE[1][1]}
    assert LOW_RELIABILITY_OUTLINE_COLOR not in scale_ends
    from housing_analyzer.map import NORMAL_OUTLINE_COLOR

    assert NORMAL_OUTLINE_COLOR not in scale_ends


def test_choropleth_layout_leaves_room_for_toolbar_and_colorbar(
    sample_prices_frame, sample_boundaries, cpi_df
):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "all",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    fig = build_choropleth_figure(frame, sample_boundaries, METRIC_PRICE)
    assert fig.layout.margin.t >= 30
    assert fig.layout.margin.t >= MAP_TOP_MARGIN
    value_traces = _value_layer_traces(fig)
    assert value_traces
    colorbar = value_traces[0].colorbar
    assert colorbar.y <= 0.2
    assert colorbar.len <= 0.85
    pct_bar = value_colorbar(METRIC_CHANGE_1Y, "1-year change (%)", -12.0, 12.0)
    assert 0.0 in pct_bar["tickvals"]


def test_plotly_map_chart_config_trims_toolbar():
    config = plotly_map_chart_config()
    assert config["displaylogo"] is False
    removed = set(config["modeBarButtonsToRemove"])
    assert {"select2d", "lasso2d", "autoScale2d"}.issubset(removed)


def test_build_choropleth_figure_returns_figure(
    sample_prices_frame, sample_boundaries, cpi_df
):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "all",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    fig = build_choropleth_figure(frame, sample_boundaries, METRIC_PRICE)
    assert fig.data
    trace_names = {trace.name for trace in fig.data}
    assert "No data" in trace_names or "Areas with data" in trace_names


def test_color_range_ignores_nan(sample_prices_frame, sample_boundaries, cpi_df):
    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "all",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    values = frame.loc[~frame["missing"], METRIC_PRICE]
    low, high = metric_color_range(values, METRIC_PRICE)
    assert not np.isnan(low)
    assert not np.isnan(high)
    assert low <= high


def test_prepare_map_dataframe_does_not_compute_per_area_summaries(
    sample_prices_frame, sample_boundaries, cpi_df, monkeypatch
):
    """The map must use the all-areas code path, not one full ranking per area."""
    import housing_analyzer.analysis.metrics as metrics

    def boom(*args, **kwargs):
        raise AssertionError("per-area summary used while preparing the map")

    monkeypatch.setattr(metrics, "summarize_area", boom)
    monkeypatch.setattr(metrics, "rank_percentile", boom)
    frame = prepare_map_dataframe(
        sample_prices_frame, sample_boundaries, "2024Q4", "all", METRIC_PRICE, cpi_df=cpi_df
    )
    assert not frame.empty


def test_prepare_map_dataframe_matches_per_area_values(
    sample_prices_frame, sample_boundaries, cpi_df
):
    from housing_analyzer.analysis.metrics import summarize_area
    from housing_analyzer.map import resolve_building_type_label

    for code in ("all", "1", "2"):
        frame = prepare_map_dataframe(
            sample_prices_frame,
            sample_boundaries,
            "2024Q4",
            code,
            METRIC_PRICE,
            cpi_df=cpi_df,
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
    sample_prices_frame, sample_boundaries, cpi_df
):
    frame = prepare_map_dataframe(
        sample_prices_frame, sample_boundaries, "2024Q4", "all", METRIC_PRICE, cpi_df=cpi_df
    )
    fig = build_choropleth_figure(frame, sample_boundaries, METRIC_PRICE)
    for trace in fig.data:
        codes = {str(code).zfill(5) for code in trace.locations}
        feature_codes = {
            str(feature["properties"]["postal_code"]).zfill(5)
            for feature in trace.geojson["features"]
        }
        assert feature_codes == codes


def test_prepare_map_dataframe_real_change_matches_hand_calculation(
    sample_prices_frame, sample_boundaries
):
    """1y real change uses CPI-deflated prices, not the raw nominal ratio."""
    prior_year_row = sample_prices_frame.loc[
        (sample_prices_frame["postal_code"] == "00100")
        & (sample_prices_frame["building_type"] == "1 — Blocks of flats, one-room flat")
        & (sample_prices_frame["quarter"] == "2024Q4")
    ].copy()
    assert len(prior_year_row) == 1
    prior_year_row["quarter"] = "2023Q4"
    prior_year_row["price_per_sqm"] = 7000.0
    extended = pd.concat([sample_prices_frame, prior_year_row], ignore_index=True)

    # CPI: 100 in 2023Q4, 104 in 2024Q4 (base quarter) -> 4% inflation.
    cpi = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2023-10-01",
                    "2023-11-01",
                    "2023-12-01",
                    "2024-10-01",
                    "2024-11-01",
                    "2024-12-01",
                ]
            ),
            "cpi": [100.0, 100.0, 100.0, 104.0, 104.0, 104.0],
        }
    )

    frame = prepare_map_dataframe(
        extended,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_CHANGE_1Y_REAL,
        cpi_df=cpi,
    )
    row = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    now_price = 7590.0
    prior_real = 7000.0 * (104.0 / 100.0)
    expected_pct = (now_price / prior_real - 1.0) * 100.0
    assert row[METRIC_CHANGE_1Y_REAL] == pytest.approx(expected_pct)
    assert not bool(row["missing"])


def test_hover_notes_cpi_not_final_for_incomplete_quarter(
    sample_prices_frame, sample_boundaries
):
    """Real-change metrics on a quarter with no final CPI shouldn't blame missing sales."""
    # Only one month of 2024Q4 is present, so cpi_by_quarter drops it as incomplete.
    cpi = pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2023-10-01", "2023-11-01", "2023-12-01", "2024-10-01"]
            ),
            "cpi": [100.0, 100.0, 100.0, 104.0],
        }
    )

    frame = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_CHANGE_1Y_REAL,
        cpi_df=cpi,
    )
    row = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    assert bool(row["missing"])
    assert NO_CPI_HOVER in row["hover"]
    assert NO_DATA_HOVER not in row["hover"]
