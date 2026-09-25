"""Housing price datasets."""

from housing_analyzer.data.boundaries import join_prices_to_areas, load_boundaries
from housing_analyzer.data.cpi import fetch_cpi, load_cpi
from housing_analyzer.data.prices import fetch_prices, load_prices

__all__ = [
    "fetch_cpi",
    "fetch_prices",
    "join_prices_to_areas",
    "load_boundaries",
    "load_cpi",
    "load_manifest",
    "load_prices",
]


def __getattr__(name: str):
    if name == "load_manifest":
        from housing_analyzer.data.snapshot import load_manifest

        return load_manifest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
