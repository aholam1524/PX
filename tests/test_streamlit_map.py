"""AppTest checks for map captions and colour-range control."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT / "app" / "streamlit_app.py"


def _caption_text(at) -> str:
    return " ".join(getattr(el, "value", "") or "" for el in at.caption)


def test_apptest_map_captions_and_full_range_checkbox(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    checkbox_labels = [cb.label or "" for cb in at.checkbox]
    assert any("Use the full value range" in label for label in checkbox_labels)

    captions = _caption_text(at)
    assert " areas have a published price for this selection." in captions
    assert "Statistics Finland publishes prices only for areas with enough sales" in captions
