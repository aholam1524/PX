"""AppTest checks for map captions and colour-range control."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from housing_analyzer.map import METRIC_CHOICES, METRIC_PRICE

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
    assert "Colour scale is fixed at 0 to 4,000 EUR/m²" in captions


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


def test_apptest_map_and_compare_single_typeahead_search(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    if not at.tabs:
        pytest.skip("AppTest tabs not available in this Streamlit version")

    at.tabs[0].run(timeout=60)
    assert not at.exception
    map_search = [sb for sb in at.selectbox if sb.key == "map_area_search"]
    assert len(map_search) == 1
    assert not [ti for ti in at.text_input if ti.key == "compare_search"]
    show_buttons = [b.label or "" for b in at.button if b.label == "Show selected area"]
    assert not show_buttons

    at.tabs[2].run(timeout=60)
    assert not at.exception
    compare_ms = [ms for ms in at.multiselect if ms.key == "compare_multiselect"]
    assert len(compare_ms) == 1
    assert compare_ms[0].max_selections == 4
    assert not [ti for ti in at.text_input if ti.key == "compare_search"]
    map_search = [sb for sb in at.selectbox if sb.key == "map_area_search"]
    assert len(map_search) == 1
    map_search[0].set_value("01200 — Hakunila (Vantaa)").run(timeout=60)
    assert not at.exception
    assert at.session_state["selected_postal_code"] == "01200"
    body = " ".join(
        getattr(el, "value", "") or ""
        for group in (at.markdown, at.subheader, at.metric)
        for el in group
    )
    assert "01200" in body


def test_apptest_map_click_syncs_search_box_and_ignores_out_of_range_code(monkeypatch):
    """Clicking the map (simulated via ``selected_postal_code``) should update the
    "Find an area" select box, but must not force it to a code that isn't one of
    its own options (e.g. a postal code with no boundary geometry on the map)."""
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    if not at.tabs:
        pytest.skip("AppTest tabs not available in this Streamlit version")

    at.tabs[0].run(timeout=60)
    assert not at.exception
    assert at.session_state["map_area_search"] == "00100"

    # Simulate a map click on "01200", which has boundary geometry and is one of
    # the "Find an area" select box's own options: the box should follow the click.
    at.session_state["selected_postal_code"] = "01200"
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["map_area_search"] == "01200"

    # Simulate a map click on "00120", which has price data but no boundary
    # geometry in the fixtures, so it is absent from the select box's options.
    # The box must keep its previous value instead of erroring or resetting.
    at.session_state["selected_postal_code"] = "00120"
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["map_area_search"] == "01200"


def test_apptest_price_layer_full_range_checkbox(monkeypatch):
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
    metric_select.set_value(METRIC_PRICE).run(timeout=60)
    assert not at.exception
    assert "Colour scale is fixed at 0 to 4,000 EUR/m²" in _caption_text(at)

    full_range = next(
        cb for cb in at.checkbox if "Use the full value range" in (cb.label or "")
    )
    full_range.set_value(True).run(timeout=60)
    assert not at.exception
    captions = _caption_text(at)
    assert "Colour scale is fixed at 0 to 4,000 EUR/m²" not in captions
