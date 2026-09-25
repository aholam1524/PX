"""AppTest smoke test for the area detail panel with fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = ROOT / "app" / "streamlit_app.py"


def test_apptest_detail_panel_with_fixtures(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    body = " ".join(
        getattr(el, "value", "") or ""
        for group in (at.markdown, at.info, at.subheader)
        for el in group
    )
    assert "Area detail" in body or any(
        "detail panel" in (getattr(el, "value", "") or "").lower() for el in at.info
    )

    if at.text_input:
        at.text_input[0].set_value("00100").run(timeout=60)
    if at.button:
        for button in at.button:
            if "Show selected area" in (button.label or ""):
                button.click().run(timeout=60)
                break

    at.run(timeout=60)
    assert not at.exception
    chunks = [
        getattr(el, "value", "") or ""
        for group in (at.subheader, at.markdown, at.metric)
        for el in group
    ]
    joined = " ".join(chunks)
    assert "00100" in joined or "Area detail" in joined


_DETAIL_PANEL_METRIC_LABELS = frozenset(
    {
        "Price per m² (EUR)",
        "1-year change",
        "5-year change",
        "Sales (4 quarters)",
        "Rank",
        "Percentile",
    }
)


def test_apptest_detail_panel_metrics_not_truncated(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.session_state["selected_postal_code"] = "00100"
    at.run(timeout=60)
    assert not at.exception

    panel_metrics = [
        m
        for m in at.metric
        if (m.label or "") in _DETAIL_PANEL_METRIC_LABELS
    ]
    assert len(panel_metrics) == 6

    for metric in panel_metrics:
        label = metric.label or ""
        value = str(metric.value or "")
        assert len(label) <= 20, f"label too long: {label!r}"
        assert len(value) <= 12, f"value too long: {value!r}"

    header_md = " ".join(getattr(el, "value", "") or "" for el in at.markdown)
    assert "· (" not in header_md


def test_apptest_relationships_tab_with_fixtures(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    if not at.tabs or len(at.tabs) < 4:
        pytest.skip("Relationships tab not available in this AppTest version")

    at.tabs[3].run(timeout=60)
    assert not at.exception

    body = " ".join(getattr(el, "value", "") or "" for el in at.subheader)
    assert "Relationships" in body


def test_apptest_compare_tab_with_fixtures(monkeypatch):
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

    at.tabs[2].run(timeout=60)
    assert not at.exception

    body = " ".join(getattr(el, "value", "") or "" for el in at.subheader)
    assert "Compare" in body or any(
        "Compare areas" in (getattr(el, "value", "") or "") for el in at.subheader
    )


def test_apptest_similar_for_selection_survives_deselecting_area(monkeypatch):
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

    at.tabs[2].run(timeout=60)
    assert not at.exception

    multiselects = [ms for ms in at.multiselect if ms.key == "compare_multiselect"]
    if not multiselects:
        pytest.skip("compare multiselect not available in this Streamlit version")

    multiselects[0].set_value(["00100", "00120"]).run(timeout=60)
    assert not at.exception

    similar_boxes = [sb for sb in at.selectbox if sb.key == "similar_for"]
    assert similar_boxes
    similar_boxes[0].set_value("00120").run(timeout=60)
    assert not at.exception
    assert [sb for sb in at.selectbox if sb.key == "similar_for"][0].value == "00120"

    # Deselecting the area that "similar_for" points to must not crash the app,
    # and the widget must fall back to an area still in the comparison.
    compare_ms = [ms for ms in at.multiselect if ms.key == "compare_multiselect"][0]
    compare_ms.set_value(["00100"]).run(timeout=60)
    assert not at.exception

    similar_box = [sb for sb in at.selectbox if sb.key == "similar_for"][0]
    assert similar_box.value == "00100"


def test_apptest_add_to_compare_duplicate_does_not_show_success(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    if at.text_input:
        at.text_input[0].set_value("00100").run(timeout=60)
    if at.button:
        for button in at.button:
            if "Show selected area" in (button.label or ""):
                button.click().run(timeout=60)
                break

    add_buttons = [b for b in at.button if (b.key or "").startswith("add_compare_")]
    if not add_buttons:
        pytest.skip("Add to comparison button not available in this Streamlit version")

    add_buttons[0].click().run(timeout=60)
    assert not at.exception
    assert any("added to comparison" in (getattr(el, "value", "") or "") for el in at.success)
    assert at.session_state["compare_postal_codes"] == ["00100"]

    # Clicking again while already in the comparison list must not show a second
    # (misleading) success toast, and must not add a duplicate entry.
    add_buttons = [b for b in at.button if (b.key or "").startswith("add_compare_")]
    add_buttons[0].click().run(timeout=60)
    assert not at.exception
    assert not any("added to comparison" in (getattr(el, "value", "") or "") for el in at.success)
    assert at.session_state["compare_postal_codes"] == ["00100"]


def test_apptest_affordability_tab_with_fixtures(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    if not at.tabs or len(at.tabs) < 2:
        pytest.skip("Affordability tab not available in this AppTest version")

    at.tabs[1].run(timeout=60)
    assert not at.exception

    body = " ".join(
        getattr(el, "value", "") or ""
        for group in (at.subheader, at.info, at.metric)
        for el in group
    )
    assert "Affordability" in body
    assert "illustration" in body.lower() or "not financial advice" in body.lower()
