"""Tests for committed data snapshots and loader snapshot preference."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from housing_analyzer.data import boundaries as boundaries_mod
from housing_analyzer.data import prices as prices_mod
from housing_analyzer.data.boundaries import load_boundaries
from housing_analyzer.data import paths as data_paths
from housing_analyzer.data.municipalities import parse_municipality_json_stat2
from housing_analyzer.data.paths import MAX_SNAPSHOT_TOTAL_BYTES
from housing_analyzer.data.prices import load_prices, parse_json_stat2
from housing_analyzer.data.snapshot import (
    build_manifest,
    check_snapshot_size,
    load_manifest,
    write_snapshot,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PRICES_SAMPLE = FIXTURES / "prices_sample.json"
BOUNDARIES_SAMPLE = FIXTURES / "boundaries_sample.geojson"
MUNICIPALITY_PRICES_SAMPLE = FIXTURES / "municipality_prices_sample.json"
MUNICIPALITY_BOUNDARIES_SAMPLE = FIXTURES / "municipality_boundaries_sample.geojson"


@pytest.fixture
def snapshot_dir(tmp_path, monkeypatch):
    snap = tmp_path / "snapshot"
    snap.mkdir()
    monkeypatch.setattr(data_paths, "SNAPSHOT_DIR", snap)
    monkeypatch.setattr(data_paths, "PRICES_SNAPSHOT_FILE", snap / "prices.csv.gz")
    monkeypatch.setattr(
        data_paths, "BOUNDARIES_SNAPSHOT_FILE", snap / "boundaries.geojson.gz"
    )
    monkeypatch.setattr(data_paths, "MANIFEST_FILE", snap / "manifest.json")
    monkeypatch.setattr(data_paths, "CPI_SNAPSHOT_FILE", snap / "cpi.csv.gz")
    monkeypatch.setattr(
        data_paths, "DEMOGRAPHICS_SNAPSHOT_FILE", snap / "demographics.csv.gz"
    )
    monkeypatch.setattr(
        data_paths, "MUNICIPALITY_PRICES_SNAPSHOT_FILE", snap / "municipality_prices.csv.gz"
    )
    monkeypatch.setattr(
        data_paths,
        "MUNICIPALITY_BOUNDARIES_SNAPSHOT_FILE",
        snap / "municipalities.geojson.gz",
    )
    return snap


@pytest.fixture
def sample_prices_frame() -> pd.DataFrame:
    with PRICES_SAMPLE.open(encoding="utf-8") as handle:
        return parse_json_stat2(json.load(handle))


@pytest.fixture
def sample_boundaries() -> dict:
    with BOUNDARIES_SAMPLE.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_build_manifest_fields():
    manifest = build_manifest(
        prices_rows=10,
        prices_bytes=1000,
        boundaries_features=3,
        boundaries_bytes=2000,
        cpi_rows=5,
        cpi_bytes=100,
    )
    assert manifest["fetch_date"]
    assert manifest["total_bytes"] == 3100
    assert "demographics.csv.gz" in manifest["files"]
    assert "municipality_prices.csv.gz" in manifest["files"]
    assert "municipalities.geojson.gz" in manifest["files"]
    assert manifest["files"]["prices.csv.gz"]["rows"] == 10
    assert manifest["files"]["boundaries.geojson.gz"]["features"] == 3
    assert "boundary_edition" in manifest["files"]["boundaries.geojson.gz"]
    assert manifest["files"]["cpi.csv.gz"]["rows"] == 5


def test_check_snapshot_size_rejects_over_limit():
    manifest = build_manifest(
        prices_rows=1,
        prices_bytes=MAX_SNAPSHOT_TOTAL_BYTES,
        boundaries_features=1,
        boundaries_bytes=1,
    )
    with pytest.raises(SystemExit):
        check_snapshot_size(manifest)


@pytest.fixture
def sample_cpi_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "cpi": [100.0, 101.0, 102.0],
        }
    )


@pytest.fixture
def sample_municipality_prices_frame() -> pd.DataFrame:
    with MUNICIPALITY_PRICES_SAMPLE.open(encoding="utf-8") as handle:
        return parse_municipality_json_stat2(json.load(handle))


@pytest.fixture
def sample_municipality_boundaries() -> dict:
    with MUNICIPALITY_BOUNDARIES_SAMPLE.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def sample_demographics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "postal_code": ["00100"],
            "data_year": [2024],
            "population": [100.0],
            "median_income_eur": [40000.0],
            "share_age_65_plus": [0.1],
            "share_higher_education": [0.3],
        }
    )


def test_write_snapshot_and_load_manifest(
    snapshot_dir,
    sample_prices_frame,
    sample_boundaries,
    sample_cpi_frame,
    sample_demographics_frame,
):
    manifest = write_snapshot(
        sample_prices_frame,
        sample_boundaries,
        sample_cpi_frame,
        sample_demographics_frame,
    )
    assert data_paths.MANIFEST_FILE.is_file()
    assert data_paths.PRICES_SNAPSHOT_FILE.is_file()
    assert data_paths.BOUNDARIES_SNAPSHOT_FILE.is_file()
    assert data_paths.CPI_SNAPSHOT_FILE.is_file()
    loaded = load_manifest()
    assert loaded is not None
    assert loaded["total_bytes"] == manifest["total_bytes"]
    assert loaded["files"]["prices.csv.gz"]["rows"] == len(sample_prices_frame)


def test_write_snapshot_with_municipality_data(
    snapshot_dir,
    sample_prices_frame,
    sample_boundaries,
    sample_cpi_frame,
    sample_demographics_frame,
    sample_municipality_prices_frame,
    sample_municipality_boundaries,
):
    manifest = write_snapshot(
        sample_prices_frame,
        sample_boundaries,
        sample_cpi_frame,
        sample_demographics_frame,
        sample_municipality_prices_frame,
        sample_municipality_boundaries,
    )
    assert data_paths.MUNICIPALITY_PRICES_SNAPSHOT_FILE.is_file()
    assert data_paths.MUNICIPALITY_BOUNDARIES_SNAPSHOT_FILE.is_file()

    mun_prices_entry = manifest["files"]["municipality_prices.csv.gz"]
    assert mun_prices_entry["rows"] == len(sample_municipality_prices_frame)
    assert mun_prices_entry["years"] == sorted(
        int(y) for y in sample_municipality_prices_frame["year"].unique()
    )

    mun_boundaries_entry = manifest["files"]["municipalities.geojson.gz"]
    assert mun_boundaries_entry["features"] == len(
        sample_municipality_boundaries["features"]
    )

    loaded = load_manifest()
    assert loaded is not None
    assert (
        loaded["files"]["municipality_prices.csv.gz"]["rows"]
        == mun_prices_entry["rows"]
    )
    assert (
        loaded["files"]["municipalities.geojson.gz"]["features"]
        == mun_boundaries_entry["features"]
    )


def test_load_prices_reads_snapshot(
    snapshot_dir,
    sample_prices_frame,
    sample_boundaries,
    sample_cpi_frame,
    sample_demographics_frame,
    monkeypatch,
):
    write_snapshot(
        sample_prices_frame,
        sample_boundaries,
        sample_cpi_frame,
        sample_demographics_frame,
    )
    cache_file = snapshot_dir.parent / "housing_prices.pkl"
    monkeypatch.setattr(prices_mod, "CACHE_FILE", cache_file)
    sample_prices_frame.to_pickle(cache_file)

    def fail_fetch(**_kwargs):
        raise AssertionError("fetch_prices should not run when snapshot exists")

    monkeypatch.setattr(prices_mod, "fetch_prices", fail_fetch)
    loaded = load_prices(refresh=False)
    assert len(loaded) == len(sample_prices_frame)
    assert loaded.loc[0, "postal_code"] == sample_prices_frame.iloc[0]["postal_code"]


def test_load_prices_refresh_bypasses_snapshot(
    snapshot_dir,
    sample_prices_frame,
    sample_boundaries,
    sample_cpi_frame,
    sample_demographics_frame,
    monkeypatch,
):
    write_snapshot(
        sample_prices_frame,
        sample_boundaries,
        sample_cpi_frame,
        sample_demographics_frame,
    )

    fetched = sample_prices_frame.head(1).copy()

    def fake_fetch(**_kwargs):
        return fetched

    monkeypatch.setattr(prices_mod, "fetch_prices", fake_fetch)
    cache_file = snapshot_dir / "cache" / "housing_prices.pkl"
    cache_file.parent.mkdir(parents=True)
    monkeypatch.setattr(prices_mod, "CACHE_DIR", cache_file.parent)
    monkeypatch.setattr(prices_mod, "CACHE_FILE", cache_file)

    loaded = load_prices(refresh=True)
    assert len(loaded) == 1
    assert cache_file.is_file()


def test_load_boundaries_reads_snapshot(
    snapshot_dir,
    sample_prices_frame,
    sample_boundaries,
    sample_cpi_frame,
    sample_demographics_frame,
):
    write_snapshot(
        sample_prices_frame,
        sample_boundaries,
        sample_cpi_frame,
        sample_demographics_frame,
    )

    def fail_fetch(_url: str) -> dict:
        raise AssertionError("WFS should not run when snapshot exists")

    loaded = load_boundaries(refresh=False, fetch_geojson=fail_fetch)
    assert loaded["type"] == "FeatureCollection"
    assert len(loaded["features"]) == len(sample_boundaries["features"])


def test_load_boundaries_refresh_bypasses_snapshot(
    snapshot_dir,
    sample_prices_frame,
    sample_boundaries,
    sample_cpi_frame,
    sample_demographics_frame,
    monkeypatch,
):
    write_snapshot(
        sample_prices_frame,
        sample_boundaries,
        sample_cpi_frame,
        sample_demographics_frame,
    )

    with BOUNDARIES_SAMPLE.open(encoding="utf-8") as handle:
        fetched = json.load(handle)
    fetched = {"type": "FeatureCollection", "features": fetched["features"][:1]}

    monkeypatch.setattr(
        boundaries_mod,
        "fetch_boundaries",
        lambda **kwargs: fetched,
    )
    cache_dir = snapshot_dir / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "housing_boundaries.geojson"
    edition_file = cache_dir / "housing_boundaries_edition.txt"
    monkeypatch.setattr(boundaries_mod, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(boundaries_mod, "CACHE_FILE", cache_file)
    monkeypatch.setattr(boundaries_mod, "EDITION_FILE", edition_file)

    loaded = load_boundaries(refresh=True)
    assert len(loaded["features"]) == 1
    assert cache_file.is_file()
