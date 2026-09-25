"""Tests for hybrid postal + municipality map layers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.data.municipalities import parse_municipality_json_stat2
from housing_analyzer.data.prices import parse_json_stat2
from housing_analyzer.hybrid_map import (
    GREY_TRACE_NAME,
    MUNICIPALITY_TRACE_NAME,
    POSTAL_FALLBACK_TRACE_NAME,
    POSTAL_TRACE_NAME,
    MapSelection,
    build_hybrid_choropleth_figure,
    building_type_mapping_caption,
    classify_hybrid_postal_coverage,
    format_municipality_hover_text,
    hybrid_coverage_counts,
    hybrid_metric_color_range,
    map_selection_from_event,
    municipality_building_type_for_map,
    municipality_map_card_data,
    prepare_municipality_map_dataframe,
    postal_to_municipality_codes,
)
from housing_analyzer.map import (
    BUDGET_FIT_LABELS,
    METRIC_CHANGE_1Y,
    METRIC_FITS_BUDGET,
    METRIC_PRICE,
    prepare_budget_fit_dataframe,
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


@pytest.fixture
def sample_municipality_prices() -> pd.DataFrame:
    with (FIXTURES / "municipality_prices_sample.json").open(encoding="utf-8") as handle:
        return parse_municipality_json_stat2(json.load(handle))


@pytest.fixture
def sample_municipality_boundaries() -> dict:
    with (FIXTURES / "municipality_boundaries_sample.geojson").open(
        encoding="utf-8"
    ) as handle:
        return json.load(handle)


@pytest.fixture
def cpi_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-10-01", "2024-11-01", "2024-12-01"]),
            "cpi": [102.0, 102.2, 102.4],
        }
    )


def test_municipality_building_type_mapping():
    assert municipality_building_type_for_map("all") == ("1", False)
    assert municipality_building_type_for_map("1") == ("3", True)
    assert municipality_building_type_for_map("5") == ("4", False)
    assert building_type_mapping_caption("2") is not None


def test_classify_hybrid_postal_coverage(
    sample_prices_frame, sample_boundaries, sample_municipality_prices, cpi_df
):
    postal = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    mun_df, _year = prepare_municipality_map_dataframe(
        sample_municipality_prices,
        cpi_df,
        "2024Q4",
        "1",
        METRIC_PRICE,
    )
    p2m = postal_to_municipality_codes(sample_boundaries)
    hybrid = classify_hybrid_postal_coverage(
        postal, METRIC_PRICE, p2m, mun_df
    )
    row_01200 = hybrid.loc[hybrid["postal_code"] == "01200"].iloc[0]
    assert row_01200["coverage"] == "municipality"
    counts = hybrid_coverage_counts(hybrid)
    assert counts.own_postal == 1
    assert counts.municipality_coloured == 1
    assert counts.no_data == 1


def test_hybrid_color_range_uses_both_levels(
    sample_prices_frame, sample_boundaries, sample_municipality_prices, cpi_df
):
    postal = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    mun_df, _ = prepare_municipality_map_dataframe(
        sample_municipality_prices,
        cpi_df,
        "2024Q4",
        "1",
        METRIC_PRICE,
    )
    p2m = postal_to_municipality_codes(sample_boundaries)
    hybrid = classify_hybrid_postal_coverage(postal, METRIC_PRICE, p2m, mun_df)
    clipped = hybrid_metric_color_range(hybrid, mun_df, METRIC_PRICE)
    full = hybrid_metric_color_range(hybrid, mun_df, METRIC_PRICE, use_full_range=True)
    assert full[1] >= clipped[1]
    assert clipped[0] <= clipped[1]


def test_municipality_hover_and_card(
    sample_municipality_prices, cpi_df, sample_boundaries
):
    mun_df, year = prepare_municipality_map_dataframe(
        sample_municipality_prices,
        cpi_df,
        "2024Q4",
        "1",
        METRIC_PRICE,
    )
    row = mun_df.loc[mun_df["municipality_code"] == "091"].iloc[0]
    hover = format_municipality_hover_text(row, METRIC_PRICE, year=year)
    assert "Municipality Helsinki" in hover
    assert f"annual figure {year}" in hover
    assert "Municipality average; the postal-code area has no published price" in hover

    card = municipality_map_card_data(
        sample_municipality_prices, "091", "2024Q4", "1"
    )
    assert card is not None
    assert card.municipality_name == "Helsinki"
    assert card.price_per_sqm == pytest.approx(4000.0)


def test_hybrid_figure_trace_order_and_geometry(
    sample_prices_frame,
    sample_boundaries,
    sample_municipality_prices,
    sample_municipality_boundaries,
    cpi_df,
):
    postal = prepare_map_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        METRIC_PRICE,
        cpi_df=cpi_df,
    )
    mun_df, _ = prepare_municipality_map_dataframe(
        sample_municipality_prices,
        cpi_df,
        "2024Q4",
        "1",
        METRIC_PRICE,
    )
    p2m = postal_to_municipality_codes(sample_boundaries)
    hybrid = classify_hybrid_postal_coverage(postal, METRIC_PRICE, p2m, mun_df)
    fig = build_hybrid_choropleth_figure(
        hybrid,
        mun_df,
        sample_boundaries,
        sample_municipality_boundaries,
        METRIC_PRICE,
    )
    assert fig.data[0].name == MUNICIPALITY_TRACE_NAME
    trace_names = [t.name for t in fig.data]
    assert POSTAL_TRACE_NAME in trace_names
    assert POSTAL_FALLBACK_TRACE_NAME in trace_names

    for trace in fig.data:
        if trace.locations is None or len(trace.locations) == 0:
            continue
        codes = {str(code).zfill(5) for code in trace.locations}
        if trace.name == MUNICIPALITY_TRACE_NAME:
            feature_codes = {
                str(f["properties"]["municipality_code"]).zfill(3)
                for f in trace.geojson["features"]
            }
            location_codes = {str(c).zfill(3) for c in trace.locations}
            assert feature_codes <= location_codes
            assert feature_codes
        elif trace.geojson and "postal_code" in (
            trace.geojson["features"][0]["properties"] if trace.geojson["features"] else {}
        ):
            feature_codes = {
                str(f["properties"]["postal_code"]).zfill(5)
                for f in trace.geojson["features"]
            }
            assert feature_codes == codes

    data_trace = next(t for t in fig.data if t.name == POSTAL_TRACE_NAME)
    assert all(z > 0 for z in data_trace.z)


class _Event:
    def __init__(self, selection):
        self.selection = selection


def test_map_selection_from_event_reads_level():
    event = _Event({"points": [{"customdata": ["01200", "Vantaa", "municipality"]}]})
    picked = map_selection_from_event(event)
    assert picked == MapSelection(postal_code="01200", level="municipality")


def test_hybrid_budget_figure_layers_municipality_and_postal(
    sample_prices_frame,
    sample_boundaries,
    sample_municipality_prices,
    sample_municipality_boundaries,
    cpi_df,
):
    size_sqm = 50.0
    max_affordable = 300_000.0
    postal = prepare_budget_fit_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        size_sqm,
        max_affordable,
        cpi_df=cpi_df,
    )
    mun_df, _year = prepare_municipality_map_dataframe(
        sample_municipality_prices,
        cpi_df,
        "2024Q4",
        "1",
        METRIC_FITS_BUDGET,
        size_sqm=size_sqm,
        max_affordable_price=max_affordable,
    )
    fig = build_hybrid_choropleth_figure(
        postal,
        mun_df,
        sample_boundaries,
        sample_municipality_boundaries,
        METRIC_FITS_BUDGET,
    )
    trace_names = [t.name for t in fig.data]

    # 00100 has its own price, so a postal-level budget-fit trace is drawn.
    assert any(name in trace_names for name in BUDGET_FIT_LABELS.values())
    # 01200 has no postal price but its municipality (Vantaa) does, so the
    # transparent postal-fallback overlay sits on top of the municipality fill.
    assert POSTAL_FALLBACK_TRACE_NAME in trace_names
    assert any(
        name == f"{label} (municipality)" for name in trace_names for label in BUDGET_FIT_LABELS.values()
    )
    # 99999 has no municipality mapping at all, so it falls through to "no data".
    assert GREY_TRACE_NAME in trace_names

    fallback_trace = next(t for t in fig.data if t.name == POSTAL_FALLBACK_TRACE_NAME)
    assert "01200" in list(fallback_trace.locations)
