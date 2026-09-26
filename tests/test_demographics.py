"""Tests for Paavo demographics loading, joins, and relationship metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_analyzer.analysis.relationships import (
    CORRELATION_DISCLAIMER,
    build_relationships_summary,
    pearson_correlation,
    prepare_relationships_frame,
    price_to_income_ratio,
)
from housing_analyzer.data.demographics import (
    DEMOGRAPHICS_COLUMNS,
    MEASURE_POPULATION,
    MEASURES_AGE_65_PLUS,
    _assemble_demographics,
    _assemble_national_demographics,
    attach_demographics_to_summaries,
    compute_inhabitant_share,
    compute_owner_occupied_share,
    compute_rented_share,
    compute_share_blocks_of_flats,
    compute_unemployment_rate,
    join_demographics_to_postal_codes,
    load_demographics,
    load_demographics_bundle,
    load_national_demographics,
)
from housing_analyzer.panel import build_area_profile_items


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


def test_derived_columns_hand_calculated():
    assert compute_rented_share(62, 100) == pytest.approx(0.62)
    assert compute_owner_occupied_share(38, 100) == pytest.approx(0.38)
    assert compute_unemployment_rate(20, 80) == pytest.approx(0.2)
    assert compute_inhabitant_share(120, 1000) == pytest.approx(0.12)
    assert compute_share_blocks_of_flats(300, 500) == pytest.approx(0.6)
    assert np.isnan(compute_rented_share(10, 0))
    assert np.isnan(compute_unemployment_rate(0, 0))


def test_join_demographics_mismatched_codes():
    demo = pd.DataFrame(
        {
            "postal_code": ["00100", "99999"],
            "data_year": [2024, 2024],
            "population": [100.0, 50.0],
            "median_income_eur": [40000.0, 30000.0],
            "share_age_65_plus": [0.1, 0.2],
            "share_higher_education": [0.3, 0.25],
            "households_total": [10.0, 5.0],
            "average_household_size": [1.5, 1.4],
            "average_floor_area_per_person": [30.0, 28.0],
            "households_owner_occupied": [6.0, 3.0],
            "households_rented": [4.0, 2.0],
            "median_household_income_eur": [40000.0, 30000.0],
            "dwellings": [12.0, 6.0],
            "average_floor_area_per_dwelling": [70.0, 65.0],
            "dwellings_blocks_of_flats": [6.0, 3.0],
            "dwellings_small_houses": [6.0, 3.0],
            "activity_inhabitants": [100.0, 50.0],
            "employed": [50.0, 25.0],
            "unemployed": [5.0, 2.0],
            "students": [10.0, 5.0],
            "pensioners": [20.0, 10.0],
            "rented_share": [0.4, 0.4],
            "owner_occupied_share": [0.6, 0.6],
            "unemployment_rate": [0.09, 0.074],
            "student_share": [0.1, 0.1],
            "pensioner_share": [0.2, 0.2],
            "share_blocks_of_flats": [0.5, 0.5],
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
            "households_total": [100.0],
            "average_household_size": [1.75],
            "average_floor_area_per_person": [35.0],
            "households_owner_occupied": [38.0],
            "households_rented": [62.0],
            "median_household_income_eur": [52000.0],
            "dwellings": [500.0],
            "average_floor_area_per_dwelling": [85.0],
            "dwellings_blocks_of_flats": [300.0],
            "dwellings_small_houses": [200.0],
            "activity_inhabitants": [1000.0],
            "employed": [80.0],
            "unemployed": [20.0],
            "students": [120.0],
            "pensioners": [250.0],
            "rented_share": [0.62],
            "owner_occupied_share": [0.38],
            "unemployment_rate": [0.2],
            "student_share": [0.12],
            "pensioner_share": [0.25],
            "share_blocks_of_flats": [0.6],
        }
    )
    enriched = attach_demographics_to_summaries(summaries, demo)
    assert enriched.loc["00100", "median_income_eur"] == 45000.0
    assert enriched.loc["00100", "rented_share"] == pytest.approx(0.62)
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
    frame = _assemble_demographics(
        population, empty, empty, empty, empty, empty, empty, data_year="2024"
    )

    complete = frame.loc[frame["postal_code"] == "00100"].iloc[0]
    partial = frame.loc[frame["postal_code"] == "00200"].iloc[0]
    assert complete["share_age_65_plus"] == pytest.approx(
        20.0 * len(MEASURES_AGE_65_PLUS) / 1000.0
    )
    assert np.isnan(partial["share_age_65_plus"])


def test_fixture_national_comparison_values(monkeypatch):
    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")
    bundle = load_demographics_bundle()
    national = bundle.national
    area = bundle.areas.loc[bundle.areas["postal_code"] == "00100"].iloc[0]
    assert national["rented_share"] == pytest.approx(0.33)
    assert area["rented_share"] == pytest.approx(0.62)
    items = build_area_profile_items(area, national)
    rented = next(item for item in items if item.label == "Rented households")
    assert "62%" in rented.area_text
    assert "33%" in rented.national_text


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


def test_prepare_relationships_frame_keeps_rows_missing_only_new_columns():
    """An area missing unemployment_rate/rented_share still counts for the
    pre-existing income/age/education plots; only its own dropna'd plots shrink."""
    base = {col: float("nan") for col in DEMOGRAPHICS_COLUMNS}
    complete = {
        **base,
        "postal_code": "00100",
        "median_income_eur": 30000.0,
        "share_age_65_plus": 0.2,
        "share_higher_education": 0.3,
        "unemployment_rate": 0.1,
        "rented_share": 0.4,
    }
    missing_new_cols = {
        **base,
        "postal_code": "00200",
        "median_income_eur": 32000.0,
        "share_age_65_plus": 0.25,
        "share_higher_education": 0.35,
    }
    demo = pd.DataFrame([complete, missing_new_cols])
    summaries = pd.DataFrame(
        {
            "price_per_sqm": [5000.0, 4500.0],
            "reliability": ["ok", "ok"],
        },
        index=["00100", "00200"],
    )

    frame, excluded = prepare_relationships_frame(summaries, demo)
    assert len(frame) == 2
    assert excluded == 0

    summary = build_relationships_summary(summaries, demo)
    plots_by_key = {plot.spec.key: plot for plot in summary.plots}
    assert plots_by_key["income"].n_areas == 2
    assert plots_by_key["unemployment"].n_areas == 1
    assert plots_by_key["rented"].n_areas == 1


def test_load_national_demographics_fixture(monkeypatch):
    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")
    national = load_national_demographics()
    assert national["average_household_size"] == pytest.approx(2.0)


def test_assemble_national_from_sss_long():
    households = pd.DataFrame(
        {
            "postal_code": ["SSS", "SSS"],
            "measure": ["te_taly", "te_vuok_as"],
            "value": [100.0, 33.0],
        }
    )
    empty = pd.DataFrame(columns=["postal_code", "measure", "value"])
    national = _assemble_national_demographics(
        empty, empty, empty, households, empty, empty, empty, data_year="2024"
    )
    assert national["households_total"] == pytest.approx(100.0)
    assert national["rented_share"] == pytest.approx(0.33)
