"""Tests for Statistics Finland housing price loading."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from housing_analyzer.data import prices as prices_mod
from housing_analyzer.data.prices import (
    fetch_prices,
    load_prices,
    parse_json_stat2,
    plan_fetch_queries,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_JSON = FIXTURES / "prices_sample.json"


@pytest.fixture
def sample_dataset() -> dict:
    with SAMPLE_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_parse_json_stat2_columns_types_and_missing(sample_dataset):
    frame = parse_json_stat2(sample_dataset)

    assert list(frame.columns) == [
        "postal_code",
        "area_name",
        "building_type",
        "quarter",
        "period_end",
        "price_per_sqm",
        "transactions",
    ]
    assert frame["postal_code"].dtype == object
    assert str(frame.loc[frame["postal_code"] == "01200", "postal_code"].iloc[0]) == "01200"
    assert frame["period_end"].dtype == "datetime64[ns]"
    assert pd.api.types.is_float_dtype(frame["price_per_sqm"])
    assert frame["transactions"].dtype == "Int64"

    missing_price = frame[
        (frame["quarter"] == "2024Q3")
        & (frame["postal_code"] == "00120")
        & (frame["building_type"].str.startswith("1 —"))
    ]["price_per_sqm"].iloc[0]
    assert pd.isna(missing_price)

    missing_tx = frame[
        (frame["quarter"] == "2024Q3")
        & (frame["postal_code"] == "00120")
        & (frame["building_type"].str.startswith("1 —"))
    ]["transactions"].iloc[0]
    assert pd.isna(missing_tx)


def test_parse_json_stat2_splits_postal_and_area(sample_dataset):
    frame = parse_json_stat2(sample_dataset)
    row = frame[frame["postal_code"] == "00100"].iloc[0]
    assert row["postal_code"] == "00100"
    assert "Helsinki keskusta" in row["area_name"]
    assert row["building_type"].startswith("1 — Blocks of flats, one-room flat")


def test_parse_json_stat2_both_measures(sample_dataset):
    frame = parse_json_stat2(sample_dataset)
    row = frame[
        (frame["postal_code"] == "00100")
        & (frame["quarter"] == "2024Q4")
        & (frame["building_type"].str.startswith("1 —"))
    ].iloc[0]
    assert row["price_per_sqm"] == pytest.approx(7590.0)
    assert row["transactions"] == 26


def test_plan_fetch_queries_respects_cell_limit():
    metadata = {
        "variables": [
            {"code": prices_mod.VAR_TIME, "values": [f"2020Q{i}" for i in range(1, 5)]},
            {"code": prices_mod.VAR_POSTAL, "values": [f"{i:05d}" for i in range(100)]},
            {"code": prices_mod.VAR_BUILDING, "values": ["1", "2", "3", "5"]},
            {"code": prices_mod.VAR_CONTENTS, "values": list(prices_mod.MEASURES)},
        ]
    }
    queries = plan_fetch_queries(metadata, max_cells=100)
    assert queries
    for chunk in queries:
        cells = (
            len(chunk["quarters"])
            * len(chunk["postal_codes"])
            * len(chunk["building_types"])
            * len(prices_mod.MEASURES)
        )
        assert cells <= 100
    assert sum(len(q["postal_codes"]) for q in queries) >= 100


def test_fetch_prices_uses_chunked_fake_api():
    metadata = {
        "variables": [
            {"code": prices_mod.VAR_TIME, "values": ["2024Q1", "2024Q2"]},
            {"code": prices_mod.VAR_POSTAL, "values": ["00100", "00120", "01200"]},
            {"code": prices_mod.VAR_BUILDING, "values": ["1", "2"]},
            {"code": prices_mod.VAR_CONTENTS, "values": list(prices_mod.MEASURES)},
        ]
    }
    calls: list[dict] = []

    def fake_post(_url: str, payload: dict) -> dict:
        calls.append(payload)
        with SAMPLE_JSON.open(encoding="utf-8") as handle:
            return json.load(handle)

    frame = fetch_prices(
        post_json=fake_post,
        get_metadata=lambda _url: metadata,
        pause_seconds=0,
    )
    expected_calls = len(plan_fetch_queries(metadata, max_cells=prices_mod.DEFAULT_MAX_CELLS))
    assert len(calls) == expected_calls
    assert not frame.empty


def test_load_prices_uses_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "housing_prices.pkl"
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prices_mod, "CACHE_FILE", cache_file)

    sample = pd.DataFrame(
        {
            "postal_code": ["00100"],
            "area_name": ["Helsinki keskusta"],
            "building_type": ["1 — Blocks of flats, one-room flat"],
            "quarter": ["2024Q4"],
            "period_end": pd.to_datetime(["2024-12-31"]),
            "price_per_sqm": [7590.0],
            "transactions": pd.array([26], dtype="Int64"),
        }
    )
    sample.to_pickle(cache_file)

    def fail_fetch(**_kwargs):
        raise AssertionError("fetch_prices should not run when cache exists")

    monkeypatch.setattr(prices_mod, "fetch_prices", fail_fetch)
    loaded = load_prices(refresh=False)
    assert len(loaded) == 1
    assert loaded.loc[0, "postal_code"] == "00100"
