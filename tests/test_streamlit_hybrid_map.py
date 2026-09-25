"""AppTest smoke tests for hybrid map controls."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT / "app" / "streamlit_app.py"


def test_apptest_hybrid_map_checkbox_and_missing_municipality_data_subprocess():
    """Isolated process: municipality snapshot absent, fixtures for postal data only."""
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path({str(ROOT)!r}) / "app"))
import housing_analyzer.hybrid_map as hybrid_map
hybrid_map.municipality_files_available = lambda: False
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(STREAMLIT_APP)!r})
at.run(timeout=60)
assert not at.exception, at.exception
captions = " ".join(getattr(el, "value", "") or "" for el in at.caption)
assert "Municipality fallback data is not in this checkout" in captions
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "HOUSING_USE_FIXTURES": "1"},
        timeout=90,
    )
    if result.returncode != 0:
        pytest.fail(result.stdout + result.stderr)


def test_apptest_hybrid_map_checkbox_off(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    fill_cb = next(
        cb for cb in at.checkbox if "Fill gaps with municipality values" in (cb.label or "")
    )
    fill_cb.set_value(False).run()
    assert not at.exception
