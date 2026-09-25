"""Tests for municipality price and boundary data layer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from housing_analyzer.data.municipalities import (
    MunicipalityYearChoice,
    apply_municipality_names_to_boundaries,
    build_municipality_boundaries,
    cpi_by_year,
    fetch_municipality_prices,
    load_municipality_prices,
    municipality_join_report,
    municipality_metrics,
    municipality_real_changes,
    municipality_year_for_quarter,
    parse_municipality_json_stat2,
    postal_codes_missing_municipality,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MUNICIPALITY_PRICES_SAMPLE = FIXTURES / "municipality_prices_sample.json"
MUNICIPALITY_POSTAL_SAMPLE = FIXTURES / "municipality_postal_sample.geojson"
MUNICIPALITY_BOUNDARIES_SAMPLE = FIXTURES / "municipality_boundaries_sample.geojson"
CPI_SAMPLE = FIXTURES / "cpi_sample.json"


@pytest.fixture
def sample_municipality_prices() -> pd.DataFrame:
    with MUNICIPALITY_PRICES_SAMPLE.open(encoding="utf-8") as handle:
        return parse_municipality_json_stat2(json.load(handle))


@pytest.fixture
def sample_postal_boundaries() -> dict:
    with MUNICIPALITY_POSTAL_SAMPLE.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def sample_municipality_boundaries() -> dict:
    with MUNICIPALITY_BOUNDARIES_SAMPLE.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_parse_municipality_prices_leading_zeros_and_missing(sample_municipality_prices):
    frame = sample_municipality_prices
    assert set(frame["municipality_code"]) == {"091", "092", "998"}
    assert frame["municipality_code"].str.len().eq(3).all()

    hels_2025 = frame.loc[
        (frame["municipality_code"] == "091")
        & (frame["year"] == 2025)
        & (frame["building_type"].str.startswith("3"))
    ].iloc[0]
    assert hels_2025["price_per_sqm"] == pytest.approx(4400.0)
    assert hels_2025["transactions"] == 12

    vantaa_2025 = frame.loc[
        (frame["municipality_code"] == "092") & (frame["year"] == 2025)
    ].iloc[0]
    assert pd.isna(vantaa_2025["price_per_sqm"])
    assert vantaa_2025["transactions"] == 3

    hels_2019 = frame.loc[
        (frame["municipality_code"] == "091") & (frame["year"] == 2019)
    ].iloc[0]
    assert hels_2019["transactions"] == 12


def test_fetch_municipality_prices_uses_fixture_parser(sample_municipality_prices):
    with MUNICIPALITY_PRICES_SAMPLE.open(encoding="utf-8") as handle:
        dataset = json.load(handle)

    def fake_post(_url: str, _payload: dict) -> dict:
        return dataset

    def fake_meta(_url: str) -> dict:
        return {
            "variables": [
                {"code": "timeperiod_y", "values": ["2025"]},
                {"code": "kunta_1_20150101", "values": ["091"]},
                {"code": "talotyyppi_5_20111209", "values": ["3"]},
                {"code": "contentscode", "values": list(dataset["dimension"]["contentscode"]["category"]["index"])},
            ]
        }

    fetched = fetch_municipality_prices(post_json=fake_post, get_metadata=fake_meta)
    assert len(fetched) == len(sample_municipality_prices)


def test_build_municipality_boundaries_dissolve_and_missing_report(
    sample_postal_boundaries, sample_municipality_prices
):
    missing = postal_codes_missing_municipality(sample_postal_boundaries)
    assert "88888" in missing

    built = build_municipality_boundaries(sample_postal_boundaries, tolerance=0.0)
    assert built["type"] == "FeatureCollection"
    assert len(built["features"]) == 2
    codes = {f["properties"]["municipality_code"] for f in built["features"]}
    assert codes == {"091", "092"}

    enriched = apply_municipality_names_to_boundaries(built, sample_municipality_prices)
    hels = next(
        f for f in enriched["features"] if f["properties"]["municipality_code"] == "091"
    )
    assert hels["properties"]["municipality_name"] == "Helsinki"


def test_municipality_join_report(sample_municipality_prices, sample_municipality_boundaries):
    report = municipality_join_report(
        sample_municipality_prices, sample_municipality_boundaries
    )
    assert "998" in report.only_in_prices
    assert "993" in report.only_in_boundaries


def test_municipality_year_for_quarter():
    choice = municipality_year_for_quarter("2025Q4", [2023, 2024, 2025])
    assert choice == MunicipalityYearChoice(year=2025, source="quarter_year")

    fallback = municipality_year_for_quarter("2026Q1", [2023, 2024, 2025])
    assert fallback.year == 2025
    assert fallback.source == "latest_available"


def test_municipality_metrics_changes_and_reliability(sample_municipality_prices):
    metrics = municipality_metrics(
        sample_municipality_prices, 2025, building_type="3"
    )
    hels = metrics.loc[metrics["municipality_code"] == "091"].iloc[0]
    assert hels["price_per_sqm"] == pytest.approx(4400.0)
    assert hels["pct_change_1y"] == pytest.approx(10.0)
    assert hels["pct_change_5y"] == pytest.approx((4400 / 3800 - 1) * 100)
    assert hels["reliability"] == "ok"

    vantaa = metrics.loc[metrics["municipality_code"] == "092"].iloc[0]
    assert pd.isna(vantaa["price_per_sqm"])
    assert vantaa["reliability"] == "none"

    pre_2020 = municipality_metrics(sample_municipality_prices, 2019, building_type="3")
    assert pre_2020.loc[pre_2020["municipality_code"] == "091", "reliability"].iloc[0] == "unknown"


def test_municipality_metrics_zero_prior_is_nan(sample_municipality_prices):
    frame = sample_municipality_prices.copy()
    frame.loc[
        (frame["municipality_code"] == "091") & (frame["year"] == 2020),
        ["price_per_sqm", "transactions"],
    ] = [0.0, 1]
    metrics = municipality_metrics(frame, 2025, building_type="3")
    hels = metrics.loc[metrics["municipality_code"] == "091"].iloc[0]
    assert pd.isna(hels["pct_change_5y"])


def test_municipality_real_changes_deflation(sample_municipality_prices):
    months_2024 = pd.date_range("2024-01-01", periods=12, freq="MS")
    months_2025 = pd.date_range("2025-01-01", periods=12, freq="MS")
    cpi_monthly = pd.DataFrame(
        {
            "month": pd.Index(months_2024).append(pd.Index(months_2025)),
            "cpi": [100.0] * 12 + [110.0] * 12,
        }
    )
    yearly = cpi_by_year(cpi_monthly)
    assert yearly.loc[2024] == pytest.approx(100.0)
    assert yearly.loc[2025] == pytest.approx(110.0)

    real = municipality_real_changes(
        sample_municipality_prices, cpi_monthly, 2025, building_type="3", base_year=2025
    )
    hels = real.loc[real["municipality_code"] == "091"].iloc[0]
    assert hels["real_price_per_sqm"] == pytest.approx(4400.0)
    assert hels["pct_change_1y_real"] == pytest.approx(
        (4400.0 / (4000.0 * (110.0 / 100.0)) - 1.0) * 100.0
    )


def test_load_municipality_prices_prefers_snapshot(tmp_path, monkeypatch, sample_municipality_prices):
    from housing_analyzer.data import municipalities as municipalities_mod
    from housing_analyzer.data import paths as data_paths

    snap = tmp_path / "municipality_prices.csv.gz"
    sample_municipality_prices.to_csv(snap, index=False, compression="gzip")
    monkeypatch.setattr(data_paths, "MUNICIPALITY_PRICES_SNAPSHOT_FILE", snap)
    monkeypatch.setattr(municipalities_mod, "CACHE_FILE", tmp_path / "cache.pkl")

    def fail_fetch(**_kwargs):
        raise AssertionError("PxWeb should not be called when snapshot exists")

    monkeypatch.setattr(municipalities_mod, "fetch_municipality_prices", fail_fetch)
    loaded = load_municipality_prices(refresh=False)
    assert len(loaded) == len(sample_municipality_prices)
