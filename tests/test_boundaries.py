"""Tests for postal-code boundary loading and price joins."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from housing_analyzer.data import boundaries as boundaries_mod
from housing_analyzer.data.boundaries import (
    join_prices_to_areas,
    load_boundaries,
    simplify_geojson,
)
from housing_analyzer.data.prices import parse_json_stat2

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BOUNDARIES_SAMPLE = FIXTURES / "boundaries_sample.geojson"
PRICES_SAMPLE = FIXTURES / "prices_sample.json"


@pytest.fixture
def sample_boundaries() -> dict:
    with BOUNDARIES_SAMPLE.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def sample_prices_frame() -> pd.DataFrame:
    with PRICES_SAMPLE.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    return parse_json_stat2(dataset)


def test_simplify_geojson_preserves_structure_and_leading_zeros():
    raw = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"posti_alue": "1200", "nimi": "Vantaa"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [25.0, 60.0],
                            [25.001, 60.0],
                            [25.0015, 60.0005],
                            [25.0005, 60.0008],
                            [25.0, 60.0],
                        ]
                    ],
                },
            }
        ],
    }
    simplified = simplify_geojson(raw, tolerance=0.001)
    assert simplified["type"] == "FeatureCollection"
    feature = simplified["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    ring = feature["geometry"]["coordinates"][0]
    assert len(ring) >= 3
    assert len(ring[0]) == 2
    assert feature["properties"]["postal_code"] == "01200"
    assert feature["properties"]["posti_alue"] == "01200"


def test_join_prices_to_areas_matches_and_reports_mismatches(
    sample_boundaries, sample_prices_frame
):
    report = join_prices_to_areas(
        sample_prices_frame,
        sample_boundaries,
        "2024Q4",
        building_type="1",
    )
    frame = report.frame
    assert set(frame["postal_code"]) == {"00100", "00120", "01200", "99999"}

    row_00100 = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    assert row_00100["has_boundary"]
    assert row_00100["price_per_sqm"] == pytest.approx(7590.0)
    assert row_00100["transactions"] == 26
    assert "Helsinki" in row_00100["area_name"]

    row_99999 = frame.loc[frame["postal_code"] == "99999"].iloc[0]
    assert row_99999["has_boundary"]
    assert pd.isna(row_99999["price_per_sqm"])
    assert pd.isna(row_99999["transactions"])

    row_00120 = frame.loc[frame["postal_code"] == "00120"].iloc[0]
    assert not row_00120["has_boundary"]
    assert not pd.isna(row_00120["price_per_sqm"])

    assert "00120" in report.only_in_prices
    assert "99999" in report.only_in_boundaries


def test_join_keeps_missing_prices_missing(sample_boundaries, sample_prices_frame):
    report = join_prices_to_areas(
        sample_prices_frame,
        sample_boundaries,
        "2024Q3",
        building_type="1",
    )
    row = report.frame.loc[report.frame["postal_code"] == "00120"].iloc[0]
    assert pd.isna(row["price_per_sqm"])
    assert pd.isna(row["transactions"])


def test_load_boundaries_uses_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "housing_boundaries.geojson"
    edition_file = tmp_path / "housing_boundaries_edition.txt"
    from housing_analyzer.data import paths as data_paths

    missing_snapshot = tmp_path / "no_boundaries.geojson.gz"
    monkeypatch.setattr(boundaries_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(boundaries_mod, "CACHE_FILE", cache_file)
    monkeypatch.setattr(boundaries_mod, "EDITION_FILE", edition_file)
    monkeypatch.setattr(data_paths, "BOUNDARIES_SNAPSHOT_FILE", missing_snapshot)

    with BOUNDARIES_SAMPLE.open(encoding="utf-8") as handle:
        cached = json.load(handle)
    cache_file.write_text(json.dumps(cached), encoding="utf-8")
    edition_file.write_text("postialue:pno_2022\n", encoding="utf-8")

    def fail_fetch(_url: str) -> dict:
        raise AssertionError("WFS should not be called when cache exists")

    loaded = load_boundaries(refresh=False, fetch_geojson=fail_fetch)
    assert loaded["type"] == "FeatureCollection"
    assert len(loaded["features"]) == 3
