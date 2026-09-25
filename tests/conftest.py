"""Shared pytest helpers for the housing analyzer test suite."""

from __future__ import annotations

import pytest

from housing_analyzer.map import FINLAND_CENTER, FINLAND_ZOOM


def assert_finland_map_layout(fig) -> None:
    assert fig.layout.map.center.lat == pytest.approx(FINLAND_CENTER["lat"], abs=0.5)
    assert fig.layout.map.center.lon == pytest.approx(FINLAND_CENTER["lon"], abs=0.5)
    assert fig.layout.map.zoom == pytest.approx(FINLAND_ZOOM)
    assert fig.layout.uirevision
