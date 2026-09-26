"""Tests for Statistics Finland rent loading (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.data import paths as data_paths
from housing_analyzer.data import rents as rents_mod
from housing_analyzer.data.rents import (
    municipality_numbers_missing_from_region_map,
    parse_rents_json_stat2,
    rent_area_for_municipality,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_rents_dataset() -> dict:
    with (FIXTURES / "rents_sample.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def sample_rents_frame(sample_rents_dataset) -> pd.DataFrame:
    return parse_rents_json_stat2(sample_rents_dataset)


@pytest.fixture
def sample_region_map() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "municipality_region_sample.csv", dtype=str)


@pytest.fixture
def snapshot_dir(tmp_path, monkeypatch):
    snap = tmp_path / "snapshot"
    snap.mkdir()
    monkeypatch.setattr(data_paths, "SNAPSHOT_DIR", snap)
    monkeypatch.setattr(data_paths, "RENTS_SNAPSHOT_FILE", snap / "rents.csv.gz")
    monkeypatch.setattr(
        data_paths, "MUNICIPALITY_REGION_SNAPSHOT_FILE", snap / "municipality_region.csv"
    )
    return snap


def test_parse_rents_fixture_columns_and_measures(sample_rents_frame):
    assert list(sample_rents_frame.columns) == [
        "area_code",
        "area_name",
        "funding",
        "rooms",
        "quarter",
        "rent_per_sqm",
        "rent_observations",
        "new_rent_per_sqm",
        "new_rent_observations",
    ]
    turku = sample_rents_frame[
        (sample_rents_frame["area_code"] == "853")
        & (sample_rents_frame["quarter"] == "2025Q1")
    ].iloc[0]
    assert turku["rent_per_sqm"] == pytest.approx(14.5)
    assert turku["rent_observations"] == 120
    assert turku["new_rent_per_sqm"] == pytest.approx(15.2)
    assert turku["new_rent_observations"] == 40


def test_parse_rents_missing_markers_stay_missing(sample_rents_frame):
    row = sample_rents_frame[
        (sample_rents_frame["area_code"] == "853")
        & (sample_rents_frame["quarter"] == "2025Q2")
    ].iloc[0]
    assert np.isnan(row["rent_per_sqm"])
    assert row["rent_observations"] == 80
    assert np.isnan(row["new_rent_per_sqm"])
    assert row["new_rent_observations"] == 20


def test_rent_area_for_municipality_city(sample_region_map):
    assert rent_area_for_municipality("853", region_map=sample_region_map) == "853"


def test_rent_area_for_municipality_region(sample_region_map):
    assert rent_area_for_municipality("202", region_map=sample_region_map) == "MK02"


def test_rent_area_for_municipality_greater_helsinki_without_city_row(
    sample_region_map,
):
    assert rent_area_for_municipality("235", region_map=sample_region_map) == "MK01"


def test_rent_area_for_municipality_fallback_without_region_map():
    assert rent_area_for_municipality("091", region_map=None) == "091"
    assert rent_area_for_municipality("235", region_map=None) == "pks"
    assert rent_area_for_municipality("018", region_map=None) == "msu"


def test_rent_area_for_municipality_unknown(sample_region_map):
    assert rent_area_for_municipality("999", region_map=sample_region_map) == "msu"


def test_municipality_numbers_missing_from_region_map(sample_region_map):
    missing = municipality_numbers_missing_from_region_map(
        ["091", "111", "202", "018"],
        sample_region_map,
    )
    assert missing == ("111",)


def test_load_rents_prefers_snapshot(snapshot_dir, sample_rents_frame, monkeypatch):
    sample_rents_frame.to_csv(
        snapshot_dir / "rents.csv.gz", index=False, compression="gzip"
    )
    monkeypatch.setattr(rents_mod, "CACHE_FILE", snapshot_dir / "rents.pkl")

    def fail_fetch(**_kwargs):
        raise AssertionError("fetch_rents should not run when snapshot exists")

    monkeypatch.setattr(rents_mod, "fetch_rents", fail_fetch)
    loaded = rents_mod.load_rents(refresh=False)
    assert len(loaded) == len(sample_rents_frame)
