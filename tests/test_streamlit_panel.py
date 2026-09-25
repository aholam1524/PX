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
