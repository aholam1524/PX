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
    metric_labels = " ".join(getattr(el, "label", "") or "" for el in at.metric)
    assert "Market activity" in metric_labels
    assert "per 1,000 inh." in joined


def test_apptest_market_activity_map_layer(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    from housing_analyzer.map import METRIC_MARKET_ACTIVITY

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    metric_select = next(
        sb for sb in at.selectbox if (sb.label or "").startswith("Metric layer")
    )
    metric_select.set_value(METRIC_MARKET_ACTIVITY).run(timeout=60)
    assert not at.exception


def test_apptest_area_profile_with_fixtures(monkeypatch):
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

    at.run(timeout=60)
    assert not at.exception

    body = " ".join(
        getattr(el, "value", "") or ""
        for group in (at.markdown, at.expander, at.caption)
        for el in group
    )
    assert "Area profile" in body or "Rented households" in body


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
        for group in (at.subheader, at.markdown, at.caption)
        for el in group
    )
    assert "Where can I afford to live" in body
    assert "Maximum affordable price" in body
    assert "What if rates rise" in body
    table_frames = [
        el.value
        for el in at.dataframe
        if getattr(el, "value", None) is not None and hasattr(el.value, "columns")
    ]
    assert table_frames, "Expected affordability area table in AppTest"
    assert "Household income (EUR/year)" in table_frames[0].columns


def test_apptest_affordability_map_payment_share_layer(monkeypatch):
    apptest_mod = importlib.util.find_spec("streamlit.testing.v1")
    if apptest_mod is None:
        pytest.skip("streamlit.testing.v1.AppTest not available in this Streamlit version")

    monkeypatch.setenv("HOUSING_USE_FIXTURES", "1")

    from streamlit.testing.v1 import AppTest

    from housing_analyzer.map import METRIC_PAYMENT_INCOME_SHARE

    at = AppTest.from_file(str(STREAMLIT_APP))
    at.run(timeout=60)
    assert not at.exception

    metric_boxes = [el for el in at.selectbox if el.label == "Metric layer"]
    if not metric_boxes:
        pytest.skip("Map metric selectbox not available in this AppTest version")
    metric_boxes[0].set_value(METRIC_PAYMENT_INCOME_SHARE).run(timeout=60)
    assert not at.exception


def test_apptest_affordability_tab_no_areas_fit(monkeypatch):
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

    at.number_input(key="afford_monthly_budget").set_value(1.0).run(timeout=60)
    at.tabs[1].run(timeout=60)
    assert not at.exception
    body = " ".join(getattr(el, "value", "") or "" for el in at.markdown)
    assert "No areas fit" in body


def test_apptest_affordability_tab_after_size_change(monkeypatch):
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

    at.number_input(key="afford_size_sqm").set_value(40.0).run(timeout=60)
    at.tabs[1].run(timeout=60)
    assert not at.exception
    body = " ".join(getattr(el, "value", "") or "" for el in at.markdown)
    assert "40 m²" in body
