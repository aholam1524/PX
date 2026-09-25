"""Tests for housing_analyzer.analysis metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from housing_analyzer.analysis.metrics import (
    pct_change,
    rank_percentile,
    regional_average,
    reliability,
    summarize_area,
)


def _row(
    postal_code: str,
    quarter: str,
    price: float | None,
    transactions: int | None,
    *,
    building_type: str = "1 — flat",
    area_name: str | None = None,
) -> dict:
    return {
        "postal_code": postal_code,
        "area_name": area_name or f"Area {postal_code}",
        "building_type": building_type,
        "quarter": quarter,
        "period_end": pd.Timestamp(f"{quarter[:4]}-12-31"),
        "price_per_sqm": float("nan") if price is None else price,
        "transactions": pd.NA if transactions is None else transactions,
    }


def _frame(*rows: dict) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    frame["transactions"] = frame["transactions"].astype("Int64")
    return frame


def test_pct_change_year_on_year_and_missing_endpoints():
    df = _frame(
        _row("00100", "2023Q4", 4000.0, 12),
        _row("00100", "2024Q4", 4400.0, 12),
        _row("00200", "2024Q4", 3000.0, 8),
    )
    snapshot = df.copy(deep=True)

    changes = pct_change(df, 4)
    assert changes.iloc[1] == pytest.approx(10.0)
    assert pd.isna(changes.iloc[0])
    assert pd.isna(changes.iloc[2])
    pd.testing.assert_frame_equal(df, snapshot)


def test_pct_change_missing_current_price_is_nan_not_zero():
    df = _frame(
        _row("00100", "2023Q4", 4000.0, 12),
        _row("00100", "2024Q4", None, 12),
    )
    changes = pct_change(df, 4)
    assert pd.isna(changes.iloc[1])


def test_pct_change_single_quarter_area():
    df = _frame(_row("00100", "2024Q4", 5000.0, 20))
    assert pd.isna(pct_change(df, 4).iloc[0])


def test_reliability_ok_low_none_and_unknown():
    df = _frame(
        _row("00100", "2019Q4", 3000.0, 50),
        _row("00100", "2020Q1", 3100.0, 2),
        _row("00100", "2020Q2", 3200.0, 2),
        _row("00100", "2020Q3", 3300.0, 2),
        _row("00100", "2020Q4", 3400.0, 2),
        _row("00200", "2020Q4", None, 5),
        _row("00300", "2020Q1", 2500.0, 5),
        _row("00300", "2020Q2", 2500.0, 5),
        _row("00300", "2020Q3", 2500.0, 5),
        _row("00300", "2020Q4", 2500.0, 5),
    )
    labels = reliability(df, min_transactions=10, window=4)
    by_key = {
        (r["postal_code"], r["quarter"]): labels.iloc[i]
        for i, r in df.iterrows()
    }
    assert by_key[("00100", "2019Q4")] == "unknown"
    assert by_key[("00100", "2020Q4")] == "low"
    assert by_key[("00200", "2020Q4")] == "none"
    assert by_key[("00300", "2020Q4")] == "ok"


def test_rank_percentile_ties_and_missing_prices():
    df = _frame(
        _row("00100", "2024Q4", 5000.0, 10, area_name="A"),
        _row("00200", "2024Q4", 4000.0, 10, area_name="B"),
        _row("00300", "2024Q4", 4000.0, 10, area_name="C"),
        _row("00400", "2024Q4", None, 10, area_name="D"),
    )
    ranked = rank_percentile(df, "2024Q4", building_type="1 — flat")
    row = ranked.set_index("postal_code")
    assert row.loc["00100", "rank"] == 1
    assert row.loc["00200", "rank"] == 2
    assert row.loc["00300", "rank"] == 2
    assert pd.isna(row.loc["00400", "rank"])
    assert row.loc["00100", "percentile"] == pytest.approx(100.0)
    assert row.loc["00200", "percentile"] == row.loc["00300", "percentile"]


def test_rank_percentile_weighted_and_simple_mean():
    df = _frame(
        _row("00100", "2024Q4", 6000.0, 10, building_type="A"),
        _row("00100", "2024Q4", 2000.0, 90, building_type="B"),
        _row("00200", "2024Q4", 3000.0, None, building_type="A"),
        _row("00200", "2024Q4", 5000.0, None, building_type="B"),
    )
    ranked = rank_percentile(df, "2024Q4", building_type=None)
    row = ranked.set_index("postal_code")
    assert row.loc["00100", "price_method"] == "weighted_mean"
    assert row.loc["00100", "price_per_sqm"] == pytest.approx(2400.0)
    assert row.loc["00200", "price_method"] == "simple_mean"
    assert row.loc["00200", "price_per_sqm"] == pytest.approx(4000.0)


def test_rank_percentile_one_area():
    df = _frame(_row("00100", "2024Q4", 4500.0, 12))
    ranked = rank_percentile(df, "2024Q4", building_type="1 — flat")
    assert ranked.loc[0, "rank"] == 1
    assert ranked.loc[0, "percentile"] == pytest.approx(100.0)


def test_regional_average_weighted_fallback_and_zero_weights():
    df = _frame(
        _row("00100", "2024Q4", 4000.0, 10),
        _row("00200", "2024Q4", 6000.0, 30),
        _row("00300", "2024Q4", 2000.0, 0),
        _row("00400", "2024Q4", 8000.0, None),
    )
    group = {
        "00100": "City",
        "00200": "City",
        "00300": "Other",
        "00400": "Other",
    }
    averages = regional_average(df, group, "2024Q4").set_index("group")
    assert averages.loc["City", "average_method"] == "weighted_mean"
    assert averages.loc["City", "price_per_sqm"] == pytest.approx(5500.0)
    assert averages.loc["Other", "average_method"] == "simple_mean"
    assert averages.loc["Other", "price_per_sqm"] == pytest.approx(5000.0)


def test_regional_average_respects_building_type():
    df = _frame(
        _row("00100", "2024Q4", 1000.0, 5, building_type="A"),
        _row("00100", "2024Q4", 9000.0, 5, building_type="B"),
    )
    group = {"00100": "Solo"}
    avg = regional_average(df, group, "2024Q4", building_type="A")
    assert len(avg) == 1
    assert avg.loc[0, "price_per_sqm"] == pytest.approx(1000.0)


def test_summarize_area_with_building_type():
    df = _frame(
        _row("00100", "2023Q4", 4000.0, 12),
        _row("00100", "2024Q4", 4400.0, 12),
        _row("00200", "2024Q4", 3000.0, 12),
    )
    summary = summarize_area(df, "00100", "2024Q4", building_type="1 — flat")
    assert summary["price_per_sqm"] == pytest.approx(4400.0)
    assert summary["pct_change_1y"] == pytest.approx(10.0)
    assert summary["reliability"] == "ok"
    assert summary["rank"] == 1


def test_input_dataframe_is_not_mutated():
    df = _frame(
        _row("00100", "2023Q4", 4000.0, 12),
        _row("00100", "2024Q4", 4400.0, 12),
    )
    before = df.copy(deep=True)
    pct_change(df, 4)
    reliability(df)
    rank_percentile(df, "2024Q4")
    regional_average(df, {"00100": "X"}, "2024Q4")
    summarize_area(df, "00100", "2024Q4")
    pd.testing.assert_frame_equal(df, before)
