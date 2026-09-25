"""Hosting readiness: Streamlit config and cold-start without snapshot."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / ".streamlit" / "config.toml"
STREAMLIT_APP = ROOT / "app" / "streamlit_app.py"


def test_streamlit_config_parses_and_hosting_flags():
    assert CONFIG_FILE.is_file(), "missing .streamlit/config.toml"
    with CONFIG_FILE.open("rb") as handle:
        config = tomllib.load(handle)

    assert config["server"]["headless"] is True
    assert config["browser"]["gatherUsageStats"] is False
    assert "theme" in config


def test_streamlit_app_missing_snapshot_does_not_raise(monkeypatch, tmp_path):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    from housing_analyzer.data import paths as data_paths

    empty_snap = tmp_path / "snapshot"
    empty_snap.mkdir()
    monkeypatch.setattr(data_paths, "SNAPSHOT_DIR", empty_snap)
    monkeypatch.setattr(data_paths, "PRICES_SNAPSHOT_FILE", empty_snap / "prices.csv.gz")
    monkeypatch.setattr(data_paths, "BOUNDARIES_SNAPSHOT_FILE", empty_snap / "boundaries.geojson.gz")
    monkeypatch.setattr(data_paths, "MANIFEST_FILE", empty_snap / "manifest.json")
    monkeypatch.delenv("HOUSING_USE_FIXTURES", raising=False)

    def fail_fetch(**_kwargs):
        raise AssertionError("live PxWeb fetch must not run when snapshot is missing")

    def fail_wfs(*_args, **_kwargs):
        raise AssertionError("live WFS fetch must not run when snapshot is missing")

    import housing_analyzer.data.prices as prices_mod
    import housing_analyzer.data.boundaries as boundaries_mod

    monkeypatch.setattr(prices_mod, "fetch_prices", fail_fetch)
    monkeypatch.setattr(boundaries_mod, "fetch_boundaries", fail_wfs)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=30)
    assert not at.exception
    chunks: list[str] = []
    for group in (at.error, at.warning, at.info, at.markdown):
        for el in group:
            val = getattr(el, "value", None)
            if isinstance(val, str):
                chunks.append(val)
    body = " ".join(chunks)
    assert "snapshot" in body.lower() or "Refresh housing data" in body
