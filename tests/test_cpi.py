"""Tests for CPI loading and inflation-adjusted prices (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.analysis.metrics import real_pct_change, to_real
from housing_analyzer.data.cpi import (
    CPI_BASE_YEAR,
    MEASURE_OVERALL_2015,
    cpi_by_quarter,
    latest_complete_quarter,
    parse_cpi_json_stat2,
)
from housing_analyzer.data import cpi as cpi_mod
from housing_analyzer.data import paths as data_paths

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def snapshot_dir(tmp_path, monkeypatch):
    snap = tmp_path / "snapshot"
    snap.mkdir()
    monkeypatch.setattr(data_paths, "SNAPSHOT_DIR", snap)
    monkeypatch.setattr(data_paths, "CPI_SNAPSHOT_FILE", snap / "cpi.csv.gz")
    monkeypatch.setattr(data_paths, "PRICES_SNAPSHOT_FILE", snap / "prices.csv.gz")
    monkeypatch.setattr(
        data_paths, "BOUNDARIES_SNAPSHOT_FILE", snap / "boundaries.geojson.gz"
    )
    monkeypatch.setattr(data_paths, "MANIFEST_FILE", snap / "manifest.json")
    return snap


@pytest.fixture
def sample_cpi_dataset() -> dict:
    with (FIXTURES / "cpi_sample.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def sample_cpi_frame(sample_cpi_dataset) -> pd.DataFrame:
    return parse_cpi_json_stat2(sample_cpi_dataset)


def test_parse_cpi_fixture_columns(sample_cpi_frame):
    assert list(sample_cpi_frame.columns) == ["month", "cpi"]
    assert len(sample_cpi_frame) == 11
    assert sample_cpi_frame["cpi"].notna().all()


def test_cpi_by_quarter_averages_three_months(sample_cpi_frame):
    quarterly = cpi_by_quarter(sample_cpi_frame)
    assert "2024Q1" in quarterly.index
    jan_feb_mar = sample_cpi_frame.loc[
        sample_cpi_frame["month"].between("2024-01-01", "2024-03-01"), "cpi"
    ]
    assert quarterly["2024Q1"] == pytest.approx(float(jan_feb_mar.mean()))


def test_cpi_by_quarter_drops_incomplete_latest_quarter(sample_cpi_frame):
    quarterly = cpi_by_quarter(sample_cpi_frame)
    assert "2024Q3" not in quarterly.index
    assert quarterly.index[-1] == "2024Q2"


def test_latest_complete_quarter(sample_cpi_frame):
    quarterly = cpi_by_quarter(sample_cpi_frame)
    assert latest_complete_quarter(quarterly) == "2024Q2"


def test_to_real_deflation_hand_calculated():
    cpi_q = pd.Series({"2024Q1": 100.0, "2024Q2": 110.0}, name="cpi")
    df = pd.DataFrame(
        {
            "postal_code": ["00100", "00100"],
            "building_type": ["1 — flat", "1 — flat"],
            "quarter": ["2024Q1", "2024Q2"],
            "price_per_sqm": [2000.0, 2200.0],
            "area_name": ["A", "A"],
            "period_end": pd.to_datetime(["2024-03-31", "2024-06-30"]),
            "transactions": [5, 5],
        }
    )
    real = to_real(df, cpi_q, base_quarter="2024Q2")
    assert real.loc[0, "real_price_per_sqm"] == pytest.approx(2200.0)
    assert real.loc[1, "real_price_per_sqm"] == pytest.approx(2200.0)


def test_to_real_missing_price_stays_missing():
    cpi_q = pd.Series({"2024Q1": 100.0}, name="cpi")
    df = pd.DataFrame(
        {
            "postal_code": ["00100"],
            "building_type": ["1 — flat"],
            "quarter": ["2024Q1"],
            "price_per_sqm": [float("nan")],
            "area_name": ["A"],
            "period_end": [pd.Timestamp("2024-03-31")],
            "transactions": [pd.NA],
        }
    )
    real = to_real(df, cpi_q, base_quarter="2024Q1")
    assert np.isnan(real.loc[0, "real_price_per_sqm"])


def test_to_real_default_base_is_latest_complete_quarter():
    cpi_q = pd.Series({"2023Q4": 100.0, "2024Q1": 105.0}, name="cpi")
    df = pd.DataFrame(
        {
            "postal_code": ["00100"],
            "building_type": ["1 — flat"],
            "quarter": ["2024Q1"],
            "price_per_sqm": [1050.0],
            "area_name": ["A"],
            "period_end": [pd.Timestamp("2024-03-31")],
            "transactions": [5],
        }
    )
    real = to_real(df, cpi_q)
    assert real.loc[0, "real_price_per_sqm"] == pytest.approx(1050.0)


def test_real_pct_change_matches_real_prices():
    cpi_q = pd.Series(
        {"2023Q1": 100.0, "2024Q1": 110.0},
        name="cpi",
    )
    df = pd.DataFrame(
        {
            "postal_code": ["00100", "00100"],
            "building_type": ["1 — flat", "1 — flat"],
            "quarter": ["2023Q1", "2024Q1"],
            "price_per_sqm": [1000.0, 1210.0],
            "area_name": ["A", "A"],
            "period_end": pd.to_datetime(["2023-03-31", "2024-03-31"]),
            "transactions": [5, 5],
        }
    )
    real_df = to_real(df, cpi_q, base_quarter="2024Q1")
    change = real_pct_change(real_df, 4)
    assert change.iloc[1] == pytest.approx(10.0)


def test_load_cpi_prefers_snapshot(snapshot_dir, sample_cpi_frame, monkeypatch):
    sample_cpi_frame.to_csv(
        snapshot_dir / "cpi.csv.gz", index=False, compression="gzip"
    )
    monkeypatch.setattr(cpi_mod, "CACHE_FILE", snapshot_dir / "cpi.pkl")

    def fail_fetch(**_kwargs):
        raise AssertionError("fetch_cpi should not run when snapshot exists")

    monkeypatch.setattr(cpi_mod, "fetch_cpi", fail_fetch)
    loaded = cpi_mod.load_cpi(refresh=False)
    assert len(loaded) == len(sample_cpi_frame)


def test_cpi_constants_documented_in_code():
    assert MEASURE_OVERALL_2015 == "ip_0_2015"
    assert CPI_BASE_YEAR == 2015
