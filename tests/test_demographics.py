"""Tests for Paavo demographics loading, joins, and relationship metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.analysis.relationships import (
    CORRELATION_DISCLAIMER,
    pearson_correlation,
    prepare_relationships_frame,
    price_to_income_ratio,
)
from housing_analyzer.data.demographics import (
    MEASURE_POPULATION,
    MEASURES_AGE_65_PLUS,
    _assemble_demographics,
    attach_demographics_to_summaries,
    join_demographics_to_postal_codes,
    load_demographics,
)


def test_price_to_income_ratio_hand_calculated():
    assert price_to_income_ratio(5000.0, 25000.0) == pytest.approx(0.2)
    assert np.isnan(price_to_income_ratio(float("nan"), 25000.0))
    assert np.isnan(price_to_income_ratio(5000.0, float("nan")))
    assert np.isnan(price_to_income_ratio(5000.0, 0.0))


def test_pearson_correlation_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    assert pearson_correlation(x, y) == pytest.approx(1.0)


def test_pearson_correlation_too_few_points():
    assert np.isnan(pearson_correlation([1.0], [2.0]))


def test_join_demographics_mismatched_codes():
    demo = pd.DataFrame(
        {
            "postal_code": ["00100", "99999"],
            "data_year": [2024, 2024],
            "population": [100.0, 50.0],
            "median_income_eur": [40000.0, 30000.0],
            "share_age_65_plus": [0.1, 0.2],
            "share_higher_education": [0.3, 0.25],
        }
    )
    report = join_demographics_to_postal_codes(demo, ["00100", "00200"])
    assert report.only_in_prices == ("00200",)
    assert report.only_in_demographics == ("99999",)
    row = report.frame.loc[report.frame["postal_code"] == "00100"].iloc[0]
    assert row["median_income_eur"] == 40000.0
    missing = report.frame.loc[report.frame["postal_code"] == "00200"].iloc[0]
    assert np.isnan(missing["median_income_eur"])


def test_attach_demographics_to_summaries():
    summaries = pd.DataFrame(
        {"price_per_sqm": [5000.0, 4000.0]},
        index=["00100", "00200"],
    )
    demo = pd.DataFrame(
        {
            "postal_code": ["00100"],
            "data_year": [2024],
            "population": [100.0],
            "median_income_eur": [45000.0],
            "share_age_65_plus": [0.12],
            "share_higher_education": [0.35],
        }
    )
    enriched = attach_demographics_to_summaries(summaries, demo)
    assert enriched.loc["00100", "median_income_eur"] == 45000.0
    assert np.isnan(enriched.loc["00200", "median_income_eur"])


def test_assemble_demographics_partial_age_bands_excluded():
    population = pd.DataFrame(
        {
            "postal_code": ["00100"] * (1 + len(MEASURES_AGE_65_PLUS))
            + ["00200"] * len(MEASURES_AGE_65_PLUS),
            "measure": [MEASURE_POPULATION, *MEASURES_AGE_65_PLUS]
            + [MEASURE_POPULATION, *MEASURES_AGE_65_PLUS[:-1]],
            "value": [1000.0] + [20.0] * len(MEASURES_AGE_65_PLUS)
            + [1000.0] + [20.0] * (len(MEASURES_AGE_65_PLUS) - 1),
        }
    )
    empty = pd.DataFrame(columns=["postal_code", "measure", "value"])
    frame = _assemble_demographics(population, empty, empty, data_year="2024")

    complete = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    partial = frame.loc[frame["postal_code"] == "00200"].iloc[0]
    assert complete["share_age_65_plus"] == pytest.approx(
        20.0 * len(MEASURES_AGE_65_PLUS) / 1000.0
    )
    assert np.isnan(partial["share_age_65_plus"])


def test_prepare_relationships_frame_excludes_unreliable(monkeypatch):
    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")
    demo = load_demographics()
    summaries = pd.DataFrame(
        {
            "price_per_sqm": [5000.0, 4000.0],
            "reliability": ["ok", "low"],
        },
        index=["00100", "00200"],
    )
    frame, excluded = prepare_relationships_frame(summaries, demo)
    assert len(frame) == 1
    assert excluded == 1
    assert CORRELATION_DISCLAIMER
