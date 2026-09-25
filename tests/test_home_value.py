"""Tests for My home value estimation (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from housing_analyzer.home_value import (
    estimate_home_value,
    indexed_area_chart_from_purchase,
    nearest_quarter_with_price,
)
ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT / "app" / "streamlit_app.py"


def _row(
    postal_code: str,
    quarter: str,
    price: float | None,
    transactions: int | None,
    *,
    building_type: str = "1 — flat",
) -> dict:
    return {
        "postal_code": postal_code,
        "area_name": f"Area {postal_code}",
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


@pytest.fixture
def cpi_q() -> pd.Series:
    return pd.Series(
        {
            "2018Q1": 100.0,
            "2018Q2": 101.0,
            "2024Q3": 115.0,
            "2024Q4": 116.0,
        },
        name="cpi",
    )


def test_estimate_nominal_and_real_change_hand_calculated(cpi_q):
    df = _frame(
        _row("00100", "2018Q2", 2000.0, 12),
        _row("00100", "2024Q4", 3000.0, 15),
    )
    result = estimate_home_value(
        df,
        cpi_q,
        "00100",
        "1 — flat",
        "2018Q2",
        200_000.0,
        latest_quarter="2024Q4",
    )
    assert result.enough_data
    assert result.price_index == pytest.approx(1.5)
    assert result.estimated_value == pytest.approx(300_000.0)
    assert result.nominal_change_pct == pytest.approx(50.0)
    # real: index * cpi_purchase / cpi_latest - 1 = 1.5 * 101/116 - 1
    assert result.real_change_pct == pytest.approx((1.5 * 101.0 / 116.0 - 1.0) * 100.0)
    assert result.reliability == "ok"


def test_nearest_quarter_when_purchase_quarter_missing(cpi_q):
    df = _frame(
        _row("00100", "2018Q1", None, 5),
        _row("00100", "2018Q2", 2000.0, 12),
        _row("00100", "2024Q4", 2400.0, 12),
    )
    result = estimate_home_value(
        df,
        cpi_q,
        "00100",
        "1 — flat",
        "2018Q1",
        100_000.0,
        latest_quarter="2024Q4",
        max_gap=2,
    )
    assert result.enough_data
    assert result.purchase_quarter_used == "2018Q2"
    assert any("nearest quarter" in n.lower() for n in result.notes)
    assert result.estimated_value == pytest.approx(120_000.0)


def test_both_ends_missing_returns_no_numbers(cpi_q):
    df = _frame(
        _row("00100", "2018Q2", 2000.0, 12),
        _row("00100", "2024Q4", 2400.0, 12),
    )
    result = estimate_home_value(
        df,
        cpi_q,
        "00100",
        "1 — flat",
        "2018Q1",
        100_000.0,
        latest_quarter="2024Q4",
        max_gap=0,
    )
    assert not result.enough_data
    assert result.estimated_value is None
    assert result.price_index is None


def test_max_gap_respected(cpi_q):
    df = _frame(
        _row("00100", "2018Q2", 2000.0, 12),
        _row("00100", "2024Q4", 2400.0, 12),
    )
    q, price = nearest_quarter_with_price(
        df, "00100", "2018Q1", "1 — flat", max_gap=0
    )
    assert q is None and price is None


def test_purchase_after_latest(cpi_q):
    df = _frame(_row("00100", "2024Q4", 3000.0, 12))
    result = estimate_home_value(
        df, cpi_q, "00100", "1 — flat", "2025Q1", 200_000.0, latest_quarter="2024Q4"
    )
    assert not result.enough_data
    assert any("after" in n.lower() for n in result.notes)


def test_non_positive_purchase_price(cpi_q):
    df = _frame(_row("00100", "2024Q4", 3000.0, 12))
    result = estimate_home_value(
        df, cpi_q, "00100", "1 — flat", "2024Q3", 0.0, latest_quarter="2024Q4"
    )
    assert not result.enough_data
    assert result.estimated_value is None


def test_unknown_postal_code(cpi_q):
    df = _frame(_row("00100", "2024Q4", 3000.0, 12))
    result = estimate_home_value(
        df, cpi_q, "99999", "1 — flat", "2024Q3", 200_000.0, latest_quarter="2024Q4"
    )
    assert not result.enough_data
    assert any("99999" in n for n in result.notes)


def test_estimate_with_all_building_types(cpi_q):
    df = _frame(
        _row("00100", "2018Q2", 2000.0, 12),
        _row("00100", "2024Q4", 3000.0, 15),
    )
    result = estimate_home_value(
        df,
        cpi_q,
        "00100",
        None,
        "2018Q2",
        200_000.0,
        latest_quarter="2024Q4",
    )
    assert result.enough_data
    assert result.building_type is None
    assert result.estimated_value == pytest.approx(300_000.0)
    assert result.nominal_change_pct == pytest.approx(50.0)


def test_real_change_unavailable_when_cpi_missing_for_latest_quarter(cpi_q):
    df = _frame(
        _row("00100", "2018Q2", 2000.0, 12),
        _row("00100", "2024Q4", 3000.0, 15),
    )
    cpi_missing_latest = cpi_q.drop(index="2024Q4")
    result = estimate_home_value(
        df,
        cpi_missing_latest,
        "00100",
        "1 — flat",
        "2018Q2",
        200_000.0,
        latest_quarter="2024Q4",
    )
    assert result.enough_data
    assert result.real_change_pct is None
    assert any("inflation-adjusted change is unavailable" in n.lower() for n in result.notes)


def test_indexed_area_chart_from_purchase():
    df = _frame(
        _row("00100", "2018Q2", 2000.0, 12),
        _row("00100", "2018Q3", 2200.0, 12),
        _row("00100", "2024Q4", 3000.0, 15),
    )
    fig = indexed_area_chart_from_purchase(df, "00100", "1 — flat", "2018Q2")
    trace = fig.data[0]
    assert list(trace.x) == ["2018Q2", "2018Q3", "2024Q4"]
    assert trace.y[0] == pytest.approx(100.0)
    assert trace.y[1] == pytest.approx(110.0)
    assert trace.y[2] == pytest.approx(150.0)


def test_apptest_my_home_tab_with_fixtures(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.session_state["selected_postal_code"] = "00100"
    at.run(timeout=60)
    assert not at.exception

    if not at.tabs or len(at.tabs) < 5:
        pytest.skip("My home tab not available in this AppTest version")

    at.tabs[4].run(timeout=60)
    assert not at.exception

    body = " ".join(
        getattr(el, "value", "") or ""
        for group in (at.subheader, at.info, at.metric)
        for el in group
    )
    info_text = " ".join(getattr(el, "value", "") or "" for el in at.info)
    assert "My home" in body
    assert "not a valuation" in info_text.lower()

    if at.number_input:
        for ni in at.number_input:
            key = ni.key or ""
            if key == "my_home_purchase_price":
                ni.set_value(200_000.0).run(timeout=60)
                break

    at.run(timeout=60)
    assert not at.exception
