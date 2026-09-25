"""Tests for the all-areas analysis functions used by the map.

The map needs a summary for every postal-code area. ``summarize_areas`` computes
them in one pass; these tests check it against the per-area ``summarize_area``
on random data with gaps, so the two can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.analysis.metrics import (
    area_prices_at,
    quarter_index,
    rank_percentile,
    reliability_at,
    shift_quarter,
    summarize_area,
    summarize_areas,
)

TYPES = ["1 — One-room flats", "2 — Two-room flats", "3 — Three-plus-room flats"]


def _random_frame(seed: int = 7) -> pd.DataFrame:
    """Synthetic prices with gaps, missing pre-2020 counts and several building types."""
    rng = np.random.default_rng(seed)
    quarters = [f"{year}Q{q}" for year in range(2018, 2025) for q in range(1, 5)]
    rows = []
    for number in range(14):
        postal = f"{number * 700 + 100:05d}"
        for building_type in TYPES:
            for quarter in quarters:
                if rng.random() < 0.08:
                    continue  # no row at all for this key
                price = float("nan") if rng.random() < 0.15 else float(rng.uniform(1500, 6000))
                pre_2020 = int(quarter[:4]) < 2020
                if pre_2020 or rng.random() < 0.1:
                    tx = pd.NA
                else:
                    tx = int(rng.integers(0, 12))
                rows.append(
                    {
                        "postal_code": postal,
                        "area_name": f"Area {postal}",
                        "building_type": building_type,
                        "quarter": quarter,
                        "period_end": pd.Timestamp(f"{quarter[:4]}-12-31"),
                        "price_per_sqm": price,
                        "transactions": tx,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["transactions"] = frame["transactions"].astype("Int64")
    return frame


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return float(a) == pytest.approx(float(b))


@pytest.mark.parametrize(
    "quarter,building_type",
    [
        ("2024Q4", None),
        ("2024Q4", TYPES[1]),
        ("2021Q2", None),
        ("2021Q2", TYPES[0]),
        ("2019Q3", None),
        ("2019Q3", TYPES[2]),
    ],
)
def test_summarize_areas_matches_per_area_summaries(quarter, building_type):
    df = _random_frame()
    bulk = summarize_areas(df, quarter, building_type=building_type)
    assert bulk.index.name == "postal_code"

    for postal in sorted(df["postal_code"].unique()):
        single = summarize_area(df, postal, quarter, building_type=building_type)
        if postal not in bulk.index:
            assert np.isnan(single["price_per_sqm"])
            assert single["reliability"] is None
            continue
        row = bulk.loc[postal]
        for key in ("price_per_sqm", "pct_change_1y", "pct_change_5y", "percentile"):
            assert _same(row[key], single[key]), (postal, key, row[key], single[key])
        assert _same(row["reliability"], single["reliability"]), (postal, "reliability")
        assert _same(row["rank"], single["rank"]), (postal, "rank")
        assert _same(row["price_method"], single["price_method"]), (postal, "method")


def test_area_prices_at_matches_rank_percentile_prices():
    df = _random_frame(seed=3)
    for building_type in (None, TYPES[1]):
        prices = area_prices_at(df, "2023Q2", building_type)
        ranked = rank_percentile(df, "2023Q2", building_type).set_index("postal_code")
        assert list(prices.index) == list(ranked.index)
        assert np.allclose(
            prices["price_per_sqm"].to_numpy(dtype=float),
            ranked["price_per_sqm"].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_rank_percentile_unknown_building_type_is_empty_not_an_error():
    df = _random_frame(seed=1)
    out = rank_percentile(df, "2023Q2", "9 — nothing")
    assert out.empty
    assert list(out.columns) == [
        "postal_code",
        "area_name",
        "price_per_sqm",
        "price_method",
        "rank",
        "percentile",
    ]


def test_reliability_at_pre_2020_is_unknown_and_validates_arguments():
    df = _random_frame(seed=2)
    labels = reliability_at(df, "2019Q1", None)
    assert set(labels.unique()) == {"unknown"}
    with pytest.raises(ValueError):
        reliability_at(df, "2023Q1", None, min_transactions=0)
    with pytest.raises(ValueError):
        reliability_at(df, "2023Q1", None, window=0)


def test_summarize_areas_unknown_quarter_returns_empty_frame():
    df = _random_frame(seed=4)
    out = summarize_areas(df, "2031Q1")
    assert out.empty
    assert "price_per_sqm" in out.columns


def test_public_quarter_helpers():
    assert quarter_index("2025Q1") == quarter_index("2024Q4") + 1
    assert shift_quarter("2025Q1", 1) == "2024Q4"
    assert shift_quarter("2025Q4", 20) == "2020Q4"
