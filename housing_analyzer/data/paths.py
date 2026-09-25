"""Shared data directory paths."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = _REPO_ROOT / "data" / "raw"
SNAPSHOT_DIR = _REPO_ROOT / "data" / "snapshot"

PRICES_SNAPSHOT_FILE = SNAPSHOT_DIR / "prices.csv.gz"
BOUNDARIES_SNAPSHOT_FILE = SNAPSHOT_DIR / "boundaries.geojson.gz"
MANIFEST_FILE = SNAPSHOT_DIR / "manifest.json"

MAX_SNAPSHOT_TOTAL_BYTES = 20 * 1024 * 1024
