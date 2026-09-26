"""Tests for mortgage affordability helpers and budget map layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.affordability import (
    BUDGET_FIT_OVER,
    BUDGET_FIT_STRETCH,
    BUDGET_FIT_WITHIN,
    affordability_summary,
    affordability_table,
    budget_ratio_and_category,
    classify_budget_fit_ratio,
    filter_affordability_table,
    loan_amount,
    max_price,
    monthly_payment,
    price_to_budget_ratio,
    sort_affordability_table,
    total_interest,
    typical_dwelling_price,
)
from housing_analyzer.data.prices import parse_json_stat2
from housing_analyzer.map import (
    METRIC_FITS_BUDGET,
    build_choropleth_figure,
    format_budget_fit_hover,
    prepare_budget_fit_dataframe,
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
    return pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-10-01", "2024-11-01", "2024-12-01"]),
            "cpi": [102.0, 102.2, 102.4],
        }
    )


def test_monthly_payment_zero_rate_thirty_years():
    principal = 360_000.0
    assert monthly_payment(principal, 0.0, 30) == pytest.approx(1000.0)


def test_monthly_payment_zero_rate_one_month_term():
    principal = 5000.0
    years = 1.0 / 12.0
    assert monthly_payment(principal, 0.0, years) == pytest.approx(5000.0)


def test_monthly_payment_six_percent_thirty_years():
    principal = 100_000.0
    payment = monthly_payment(principal, 6.0, 30)
    assert payment == pytest.approx(599.55, rel=1e-3)


def test_total_interest_matches_payment_sum():
    principal = 100_000.0
    years = 30
    payment = monthly_payment(principal, 6.0, years)
    interest = total_interest(principal, 6.0, years)
    assert interest == pytest.approx(payment * 360 - principal, rel=1e-6)


def test_loan_amount_and_validation():
    assert loan_amount(200_000.0, 50_000.0) == pytest.approx(150_000.0)
    with pytest.raises(ValueError, match="down payment"):
        loan_amount(100_000.0, 100_001.0)
    with pytest.raises(ValueError, match="negative"):
        loan_amount(-1.0, 0.0)


def test_max_price_round_trip_with_monthly_payment():
    budget = 1200.0
    down = 40_000.0
    rate = 4.5
    years = 25.0
    cap = max_price(budget, down, rate, years)
    principal = loan_amount(cap, down)
    payment = monthly_payment(principal, rate, years)
    assert payment == pytest.approx(budget, rel=1e-6)
    assert payment <= budget + 1e-6


def test_max_price_zero_rate():
    budget = 800.0
    down = 10_000.0
    years = 20.0
    cap = max_price(budget, down, 0.0, years)
    principal = loan_amount(cap, down)
    assert monthly_payment(principal, 0.0, years) == pytest.approx(budget)


def test_validation_non_positive_term():
    with pytest.raises(ValueError, match="positive"):
        monthly_payment(1000.0, 3.0, 0.0)


def test_classify_budget_fit_boundaries():
    assert classify_budget_fit_ratio(1.0) == BUDGET_FIT_WITHIN
    assert classify_budget_fit_ratio(0.5) == BUDGET_FIT_WITHIN
    assert classify_budget_fit_ratio(1.2) == BUDGET_FIT_STRETCH
    assert classify_budget_fit_ratio(1.2000001) == BUDGET_FIT_OVER
    assert classify_budget_fit_ratio(2.0) == BUDGET_FIT_OVER


def test_price_to_budget_ratio():
    assert price_to_budget_ratio(200_000.0, 250_000.0) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        price_to_budget_ratio(100.0, 0.0)


def test_prepare_budget_fit_dataframe_colours_and_missing(
    sample_prices_frame, sample_boundaries, cpi_df
):
    size_sqm = 50.0
    max_affordable = 300_000.0
    frame = prepare_budget_fit_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        size_sqm,
        max_affordable,
        cpi_df=cpi_df,
    )
    row_ok = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    assert row_ok["budget_fit"] in {
        BUDGET_FIT_WITHIN,
        BUDGET_FIT_STRETCH,
        BUDGET_FIT_OVER,
    }
    assert not bool(row_ok["missing"])

    missing_row = frame.loc[frame["postal_code"] == "01200"].iloc[0]
    assert bool(missing_row["missing"])
    assert pd.isna(missing_row["budget_fit"])

    fig = build_choropleth_figure(frame, sample_boundaries, METRIC_FITS_BUDGET)
    assert fig.data


def test_prepare_budget_fit_dataframe_zero_budget_is_over_not_error(
    sample_prices_frame, sample_boundaries, cpi_df
):
    """A 0 EUR max affordable price (e.g. 100% down payment) is a valid input;
    priced areas should be classified as over budget rather than raising."""
    frame = prepare_budget_fit_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        50.0,
        0.0,
        cpi_df=cpi_df,
    )
    row_ok = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    assert not bool(row_ok["missing"])
    assert row_ok["budget_fit"] == BUDGET_FIT_OVER


def test_format_budget_fit_hover_zero_budget_is_over_not_error():
    row = {"postal_code": "00100", "area_name": "Test", "price_per_sqm": 5000.0}
    hover = format_budget_fit_hover(row, max_affordable_price=0.0, size_sqm=50.0)
    assert "More than 20% over budget" in hover
    assert "Max affordable: 0 EUR" in hover


def test_affordability_table_typical_price_payment_and_headroom(sample_prices_frame):
    size_sqm = 50.0
    price_sqm = 7590.0
    max_aff = 400_000.0
    table = affordability_table(
        sample_prices_frame,
        "2024Q4",
        "1",
        size_sqm,
        max_aff,
        4.0,
        25.0,
        20.0,
        True,
    )
    row = table.loc[table["postal_code"] == "00100"].iloc[0]
    typical = typical_dwelling_price(price_sqm, size_sqm)
    assert row["typical_price"] == pytest.approx(typical)
    assert row["price_per_sqm"] == pytest.approx(price_sqm)
    assert row["headroom"] == pytest.approx(max_aff - typical)
    down = typical * 0.2
    principal = loan_amount(typical, down)
    assert row["monthly_payment"] == pytest.approx(
        monthly_payment(principal, 4.0, 25.0), rel=1e-6
    )


def test_affordability_table_fixed_eur_down_payment_over_typical_price(
    sample_prices_frame,
):
    """EUR-mode down payment above a cheap area's price must not raise."""
    size_sqm = 50.0
    max_aff = 400_000.0
    table = affordability_table(
        sample_prices_frame,
        "2024Q4",
        "1",
        size_sqm,
        max_aff,
        4.0,
        25.0,
        1_000_000.0,
        False,
    )
    assert not table.empty
    assert (table["budget_fit"] == BUDGET_FIT_OVER).all()
    assert table["monthly_payment"].isna().all()


def test_budget_ratio_and_category_at_max_and_twenty_percent_over():
    max_aff = 100_000.0
    assert budget_ratio_and_category(100_000.0, max_aff) == (
        pytest.approx(1.0),
        BUDGET_FIT_WITHIN,
    )
    assert budget_ratio_and_category(120_000.0, max_aff) == (
        pytest.approx(1.2),
        BUDGET_FIT_STRETCH,
    )
    assert budget_ratio_and_category(120_001.0, max_aff)[1] == BUDGET_FIT_OVER


def test_affordability_summary_counts_and_shares(sample_prices_frame):
    table = affordability_table(
        sample_prices_frame,
        "2024Q4",
        "1",
        50.0,
        500_000.0,
        4.0,
        25.0,
        20.0,
        True,
    )
    summary = affordability_summary(table, max_affordable_price=500_000.0)
    assert summary["n_with_price"] == 2
    assert summary["n_fits"] == 2
    assert summary["fit_share_pct"] == pytest.approx(100.0)

    none_fit = affordability_table(
        sample_prices_frame,
        "2024Q4",
        "1",
        60.0,
        200_000.0,
        4.0,
        25.0,
        20.0,
        True,
    )
    none_summary = affordability_summary(none_fit, max_affordable_price=200_000.0)
    assert none_summary["n_fits"] == 0
    assert none_summary["fit_share_pct"] == pytest.approx(0.0)


def test_filter_and_sort_affordability_table(sample_prices_frame):
    table = affordability_table(
        sample_prices_frame,
        "2024Q4",
        "1",
        50.0,
        500_000.0,
        4.0,
        25.0,
        20.0,
        True,
    )
    only = filter_affordability_table(table, only_fits=True)
    assert len(only) == 2
    assert set(only["budget_fit"]) == {BUDGET_FIT_WITHIN}

    helsinki = filter_affordability_table(
        table, municipalities=["Helsinki"]
    )
    assert set(helsinki["postal_code"]) == {"00100", "00120"}

    sorted_table = sort_affordability_table(table)
    assert sorted_table.iloc[0]["budget_fit"] == BUDGET_FIT_WITHIN
    assert sorted_table["price_per_sqm"].is_monotonic_increasing


def test_affordability_table_matches_budget_map_layer(
    sample_prices_frame, sample_boundaries, cpi_df
):
    size_sqm = 50.0
    max_aff = 300_000.0
    rate = 4.5
    years = 25.0
    table = affordability_table(
        sample_prices_frame,
        "2024Q4",
        "1",
        size_sqm,
        max_aff,
        rate,
        years,
        20.0,
        True,
    ).set_index("postal_code")
    frame = prepare_budget_fit_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "1",
        size_sqm,
        max_aff,
        annual_rate_pct=rate,
        years=years,
        down_payment_value=20.0,
        use_percent=True,
        cpi_df=cpi_df,
    )
    for _, row in frame.loc[~frame["missing"]].iterrows():
        code = str(row["postal_code"]).zfill(5)
        assert row["budget_fit"] == table.loc[code, "budget_fit"]
        assert row["budget_ratio"] == pytest.approx(
            table.loc[code, "budget_ratio"], rel=1e-9, nan_ok=True
        )


def test_classify_at_exact_budget_and_twenty_percent_over():
    max_aff = 100_000.0
    assert classify_budget_fit_ratio(price_to_budget_ratio(100_000.0, max_aff)) == (
        BUDGET_FIT_WITHIN
    )
    assert classify_budget_fit_ratio(price_to_budget_ratio(120_000.0, max_aff)) == (
        BUDGET_FIT_STRETCH
    )
    assert classify_budget_fit_ratio(price_to_budget_ratio(120_001.0, max_aff)) == (
        BUDGET_FIT_OVER
    )
