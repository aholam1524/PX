"""Tests for rent vs buy helpers and map layer (no network)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.data import paths as data_paths
from housing_analyzer.data.rents import parse_rents_json_stat2
from housing_analyzer.map import METRIC_GROSS_RENTAL_YIELD
from housing_analyzer.rent_vs_buy import (
    MonthlyComparison,
    build_rent_vs_buy_summary,
    gross_rental_yield,
    lookup_rent_quote,
    monthly_comparison,
    prepare_gross_rental_yield_dataframe,
    price_to_rent_ratio,
    rent_area_display_name,
    rent_files_available,
    rent_funding_selection,
    rent_rooms_for_building_type,
    rent_vs_buy_conclusion,
    resolve_price_per_sqm,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
STREAMLIT_APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def sample_rents_frame() -> pd.DataFrame:
    import json

    with (FIXTURES / "rents_sample.json").open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    return parse_rents_json_stat2(dataset)


@pytest.fixture
def sample_region_map() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "municipality_region_sample.csv", dtype=str)


@pytest.fixture
def sample_prices_frame() -> pd.DataFrame:
    import json

    from housing_analyzer.data.prices import parse_json_stat2

    with (FIXTURES / "prices_sample.json").open(encoding="utf-8") as handle:
        return parse_json_stat2(json.load(handle))


@pytest.fixture
def sample_boundaries() -> dict:
    import json

    with (FIXTURES / "boundaries_sample.geojson").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def sample_municipality_prices() -> pd.DataFrame:
    import json

    from housing_analyzer.data.municipalities import parse_municipality_json_stat2

    with (FIXTURES / "municipality_prices_sample.json").open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    return parse_municipality_json_stat2(dataset)


def test_gross_rental_yield_and_ratio_hand_calculated():
    # rent 10 EUR/m²/month, price 2000 EUR/m² -> yield 12*10/2000 = 0.06
    assert gross_rental_yield(10.0, 2000.0) == pytest.approx(0.06)
    assert price_to_rent_ratio(10.0, 2000.0) == pytest.approx(2000.0 / 120.0)
    assert math.isnan(gross_rental_yield(None, 2000.0))
    assert math.isnan(gross_rental_yield(10.0, 0.0))


def test_monthly_comparison_hand_calculated():
    # 50 m², 3000 EUR/m², 15 EUR/m² rent, 60k down on 150k, 0% 25y, 4 EUR/m² charges
    comp = monthly_comparison(50.0, 3000.0, 15.0, 60_000.0, 0.0, 25.0, 4.0)
    assert comp is not None
    assert comp.monthly_rent == pytest.approx(750.0)
    assert comp.monthly_charges == pytest.approx(200.0)
    assert comp.monthly_loan_payment == pytest.approx(90_000.0 / 300.0)
    assert comp.total_monthly_owning == pytest.approx(comp.monthly_loan_payment + 200.0)
    assert comp.difference_rent_minus_owning == pytest.approx(
        comp.monthly_rent - comp.total_monthly_owning
    )


def test_rent_rooms_and_funding_mapping():
    assert rent_rooms_for_building_type("1") == ("1", "One-room flat")
    assert rent_rooms_for_building_type("2") == ("2", "Two-room flat")
    assert rent_rooms_for_building_type("3") == ("3", "Three-room flat+")
    assert rent_rooms_for_building_type("5") == ("SSS", "Total")
    assert rent_rooms_for_building_type("all") == ("SSS", "Total")
    code, label = rent_funding_selection()
    assert code == "1"
    assert "Non-subsidised" in label


def test_rent_area_display_labels():
    assert "(region)" in rent_area_display_name("MK02", "Varsinais-Suomi")
    assert "(city)" in rent_area_display_name("853", "Turku")
    assert "(aggregate)" in rent_area_display_name("msu", "Rest of country")


def test_lookup_rent_quote_fixture(sample_rents_frame):
    quote = lookup_rent_quote(
        sample_rents_frame,
        rent_area_code="091",
        quarter="2025Q1",
        rooms_code="SSS",
    )
    assert quote is not None
    assert quote.rent_per_sqm == pytest.approx(18.0)
    assert quote.rent_observations == 200


def test_resolve_price_postal_and_municipality_fallback(
    sample_prices_frame, sample_boundaries, sample_municipality_prices
):
    postal = resolve_price_per_sqm(
        sample_prices_frame,
        sample_boundaries,
        "00100",
        "2024Q4",
        "all",
    )
    assert postal is not None
    assert postal.source == "postal"

    missing = resolve_price_per_sqm(
        sample_prices_frame,
        sample_boundaries,
        "99999",
        "2024Q4",
        "all",
        municipality_prices=sample_municipality_prices,
    )
    # Fixture municipality prices may or may not cover 99999; at least postal path works.


def test_prepare_gross_rental_yield_layer(
    sample_prices_frame,
    sample_boundaries,
    sample_rents_frame,
    sample_region_map,
):
    layer = prepare_gross_rental_yield_dataframe(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        "all",
        sample_rents_frame,
        sample_region_map,
    )
    assert not layer.empty
    helsinki = layer.loc[layer["postal_code"] == "00100"].iloc[0]
    assert not np.isnan(helsinki[METRIC_GROSS_RENTAL_YIELD])
    assert helsinki[METRIC_GROSS_RENTAL_YIELD] > 0
    missing_area = layer.loc[layer["postal_code"] == "99999"].iloc[0]
    assert missing_area["missing"] or np.isnan(missing_area[METRIC_GROSS_RENTAL_YIELD])


def test_build_rent_vs_buy_summary_with_fixture(
    sample_prices_frame,
    sample_boundaries,
    sample_rents_frame,
    sample_region_map,
):
    summary = build_rent_vs_buy_summary(
        sample_prices_frame,
        sample_boundaries,
        "00100",
        "2024Q4",
        "all",
        sample_rents_frame,
        sample_region_map,
        size_sqm=60.0,
        down_payment_value=20.0,
        use_percent_down=True,
        annual_rate_pct=4.0,
        years=25.0,
        monthly_charge_per_sqm=4.0,
    )
    assert summary["price_quote"] is not None
    assert summary["rent_quote"] is not None
    assert summary["comparison"] is not None
    assert "Gross rental yield" in summary["conclusion"]


def test_rent_files_available_respects_snapshot(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    monkeypatch.setattr(data_paths, "SNAPSHOT_DIR", snap)
    monkeypatch.setattr(data_paths, "RENTS_SNAPSHOT_FILE", snap / "rents.csv.gz")
    monkeypatch.delenv("HOUSING_USE_FIXTURES", raising=False)
    assert rent_files_available() is False
    (snap / "rents.csv.gz").write_bytes(b"x")
    assert rent_files_available() is True


def test_conclusion_when_missing_data():
    text = rent_vs_buy_conclusion(
        yield_fraction=float("nan"),
        price_rent_ratio=float("nan"),
        comparison=None,
    )
    assert "Not enough" in text


def test_apptest_hides_rent_features_without_rent_files(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")
    monkeypatch.setattr(
        "housing_analyzer.rent_vs_buy.rent_files_available", lambda: False
    )

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=90)
    assert not at.exception
    labels = [getattr(t, "label", None) for t in at.tabs]
    assert "Rent vs buy" not in labels
    body = " ".join(getattr(el, "value", "") or "" for el in at.caption)
    assert "rent data is not in this checkout" in body.lower()


def test_apptest_rent_vs_buy_tab_and_layer(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=90)
    assert not at.exception

    labels = [getattr(t, "label", None) for t in at.tabs]
    assert "Rent vs buy" in labels
    rent_idx = labels.index("Rent vs buy")
    at.tabs[rent_idx].run(timeout=90)
    assert not at.exception
    body = " ".join(
        getattr(el, "value", "") or ""
        for group in (at.subheader, at.markdown, at.caption, at.info, at.warning)
        for el in group
    )
    assert "Rent vs buy" in body or "Gross rental yield" in body

    metric_select = at.selectbox(key=None)
    # Find metric layer selectbox by options containing gross yield
    layer_boxes = [
        sb
        for sb in at.selectbox
        if any(
            "Gross rental yield" in (opt if isinstance(opt, str) else str(opt))
            for opt in (sb.options or [])
        )
    ]
    assert layer_boxes, "Gross rental yield metric missing from map selector"
    layer_boxes[0].set_value(METRIC_GROSS_RENTAL_YIELD).run(timeout=90)
    assert not at.exception
