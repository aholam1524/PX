"""AppTest checks for map captions and colour-range control."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from housing_analyzer.map import METRIC_CHOICES

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
    assert any("Fill gaps with municipality values" in label for label in checkbox_labels)

    captions = _caption_text(at)
    assert "postal-code areas have their own price" in captions
    assert "Municipality values are annual figures" in captions


def test_apptest_map_default_selection_and_detail_panel(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["selected_postal_code"] == "00100"

    body = " ".join(
        getattr(el, "value", "") or ""
        for group in (at.markdown, at.subheader, at.metric)
        for el in group
    )
    assert "00100" in body
    click_map_msg = "Click a map area or search"
    info_text = " ".join(getattr(el, "value", "") or "" for el in at.info)
    assert click_map_msg not in info_text

    at.session_state["selected_postal_code"] = "00120"
    at.run(timeout=60)
    assert at.session_state["selected_postal_code"] == "00120"


def test_apptest_map_renders_each_metric_layer(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    metric_select = next(
        sb for sb in at.selectbox if (sb.label or "").startswith("Metric layer")
    )
    for key, _label in METRIC_CHOICES:
        metric_select.set_value(key).run(timeout=60)
        assert not at.exception, f"Map failed for metric {key}"
