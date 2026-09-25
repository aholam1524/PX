"""Smoke tests for housing_analyzer and the Streamlit app."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from housing_analyzer import price_per_sqm

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT / "app" / "streamlit_app.py"


def test_price_per_sqm_normal():
    assert price_per_sqm(250_000, 50) == 5000.0


def test_price_per_sqm_non_positive_area():
    with pytest.raises(ValueError, match="area_sqm must be positive"):
        price_per_sqm(100_000, 0)
    with pytest.raises(ValueError, match="area_sqm must be positive"):
        price_per_sqm(100_000, -10)


def test_streamlit_app_runs_without_exception(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=30)
    assert not at.exception
