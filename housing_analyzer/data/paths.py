"""Shared data directory paths."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = _REPO_ROOT / "data" / "raw"
SNAPSHOT_DIR = _REPO_ROOT / "data" / "snapshot"

PRICES_SNAPSHOT_FILE = SNAPSHOT_DIR / "prices.csv.gz"
BOUNDARIES_SNAPSHOT_FILE = SNAPSHOT_DIR / "boundaries.geojson.gz"
CPI_SNAPSHOT_FILE = SNAPSHOT_DIR / "cpi.csv.gz"
DEMOGRAPHICS_SNAPSHOT_FILE = SNAPSHOT_DIR / "demographics.csv.gz"
MANIFEST_FILE = SNAPSHOT_DIR / "manifest.json"

MAX_SNAPSHOT_TOTAL_BYTES = 20 * 1024 * 1024

FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"
PRICES_FIXTURE_FILE = FIXTURES_DIR / "prices_sample.json"
BOUNDARIES_FIXTURE_FILE = FIXTURES_DIR / "boundaries_sample.geojson"
CPI_FIXTURE_FILE = FIXTURES_DIR / "cpi_sample.json"
DEMOGRAPHICS_FIXTURE_FILE = FIXTURES_DIR / "demographics_sample.csv"


def use_fixtures() -> bool:
    import os

    return os.environ.get("HOUSING_USE_FIXTURES") == "1"
