"""Build and read committed housing data snapshots."""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from housing_analyzer.data.boundaries import (
    EDITION_LABEL,
    FEATURE_TYPE,
    WFS_BASE,
    fetch_boundaries,
    load_boundaries,
)
from housing_analyzer.data import paths as data_paths
from housing_analyzer.data.cpi import API_URL as CPI_API_URL
from housing_analyzer.data.cpi import fetch_cpi, load_cpi
from housing_analyzer.data.demographics import (
    DEFAULT_DATA_YEAR,
    PAAVO_BASE,
    TABLE_EDUCATION,
    TABLE_INCOME,
    TABLE_POPULATION,
    fetch_demographics,
    load_demographics,
)
from housing_analyzer.data.municipalities import (
    API_URL as MUNICIPALITY_API_URL,
    apply_municipality_names_to_boundaries,
    build_municipality_boundaries,
    fetch_municipality_prices,
    load_municipality_prices,
    municipality_join_report,
)
from housing_analyzer.data.prices import API_URL, fetch_prices, load_prices

DEMOGRAPHICS_SOURCE_NOTE = (
    f"{PAAVO_BASE} — tables 12ey.px, 12f1.px, 12ez.px (year {DEFAULT_DATA_YEAR})"
)

BOUNDARIES_SOURCE_URL = (
    f"{WFS_BASE}?service=WFS&version=2.0.0&request=GetFeature"
    f"&typeName={FEATURE_TYPE}&outputFormat=application%2Fjson&srsName=EPSG%3A4326"
)


def snapshot_is_complete() -> bool:
    """Return True when all snapshot artifacts are present."""
    return (
        data_paths.PRICES_SNAPSHOT_FILE.is_file()
        and data_paths.BOUNDARIES_SNAPSHOT_FILE.is_file()
        and data_paths.CPI_SNAPSHOT_FILE.is_file()
        and data_paths.DEMOGRAPHICS_SNAPSHOT_FILE.is_file()
        and data_paths.MANIFEST_FILE.is_file()
    )


def load_manifest() -> dict[str, Any] | None:
    """Return the snapshot manifest, or None if it is missing."""
    if not data_paths.MANIFEST_FILE.is_file():
        return None
    with data_paths.MANIFEST_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def total_snapshot_bytes(manifest: dict[str, Any]) -> int:
    """Sum recorded file sizes from a manifest."""
    files = manifest.get("files")
    if not isinstance(files, dict):
        return 0
    total = 0
    for entry in files.values():
        if isinstance(entry, dict) and "bytes" in entry:
            total += int(entry["bytes"])
    return total


def check_snapshot_size(manifest: dict[str, Any]) -> int:
    """Return total bytes; raise SystemExit if above the repo limit."""
    total = total_snapshot_bytes(manifest)
    if total > data_paths.MAX_SNAPSHOT_TOTAL_BYTES:
        limit_mb = data_paths.MAX_SNAPSHOT_TOTAL_BYTES / (1024 * 1024)
        actual_mb = total / (1024 * 1024)
        raise SystemExit(
            f"Snapshot total size {actual_mb:.2f} MB exceeds limit of {limit_mb:.0f} MB"
        )
    return total


def _write_prices_snapshot(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return path.stat().st_size


def _write_cpi_snapshot(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return path.stat().st_size


def _write_demographics_snapshot(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return path.stat().st_size


def _write_municipality_prices_snapshot(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return path.stat().st_size


def _write_municipality_boundaries_snapshot(collection: dict[str, Any], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(collection, ensure_ascii=False, separators=(",", ":"))
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    return path.stat().st_size


def _write_boundaries_snapshot(collection: dict[str, Any], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(collection, ensure_ascii=False, separators=(",", ":"))
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    return path.stat().st_size


def build_manifest(
    *,
    prices_rows: int,
    prices_bytes: int,
    boundaries_features: int,
    boundaries_bytes: int,
    cpi_rows: int = 0,
    cpi_bytes: int = 0,
    demographics_rows: int = 0,
    demographics_bytes: int = 0,
    municipality_prices_rows: int = 0,
    municipality_prices_bytes: int = 0,
    municipality_prices_years: list[int] | None = None,
    municipality_boundaries_features: int = 0,
    municipality_boundaries_bytes: int = 0,
    fetch_date: date | None = None,
) -> dict[str, Any]:
    """Build manifest metadata for a snapshot directory."""
    when = fetch_date or date.today()
    files = {
        "prices.csv.gz": {
            "bytes": prices_bytes,
            "rows": prices_rows,
            "source_url": API_URL,
        },
        "boundaries.geojson.gz": {
            "bytes": boundaries_bytes,
            "features": boundaries_features,
            "source_url": BOUNDARIES_SOURCE_URL,
            "boundary_edition": EDITION_LABEL,
            "feature_type": FEATURE_TYPE,
        },
        "cpi.csv.gz": {
            "bytes": cpi_bytes,
            "rows": cpi_rows,
            "source_url": CPI_API_URL,
        },
        "demographics.csv.gz": {
            "bytes": demographics_bytes,
            "rows": demographics_rows,
            "source_url": DEMOGRAPHICS_SOURCE_NOTE,
            "paavo_tables": {
                "population": TABLE_POPULATION,
                "income": TABLE_INCOME,
                "education": TABLE_EDUCATION,
            },
            "data_year": DEFAULT_DATA_YEAR,
        },
        "municipality_prices.csv.gz": {
            "bytes": municipality_prices_bytes,
            "rows": municipality_prices_rows,
            "source_url": MUNICIPALITY_API_URL,
            "table_id": "13mx.px",
            "years": municipality_prices_years or [],
        },
        "municipalities.geojson.gz": {
            "bytes": municipality_boundaries_bytes,
            "features": municipality_boundaries_features,
            "source_url": BOUNDARIES_SOURCE_URL,
            "derived_from": "boundaries.geojson.gz",
        },
    }
    total = (
        prices_bytes
        + boundaries_bytes
        + cpi_bytes
        + demographics_bytes
        + municipality_prices_bytes
        + municipality_boundaries_bytes
    )
    return {
        "fetch_date": when.isoformat(),
        "files": files,
        "total_bytes": total,
    }


def write_snapshot(
    prices: pd.DataFrame,
    boundaries: dict[str, Any],
    cpi: pd.DataFrame,
    demographics: pd.DataFrame,
    municipality_prices: pd.DataFrame | None = None,
    municipality_boundaries: dict[str, Any] | None = None,
    *,
    fetch_date: date | None = None,
) -> dict[str, Any]:
    """Write snapshot files and manifest; enforce the size limit."""
    prices_bytes = _write_prices_snapshot(prices, data_paths.PRICES_SNAPSHOT_FILE)
    boundaries_bytes = _write_boundaries_snapshot(
        boundaries, data_paths.BOUNDARIES_SNAPSHOT_FILE
    )
    cpi_bytes = _write_cpi_snapshot(cpi, data_paths.CPI_SNAPSHOT_FILE)
    demographics_bytes = _write_demographics_snapshot(
        demographics, data_paths.DEMOGRAPHICS_SNAPSHOT_FILE
    )

    mun_prices = municipality_prices if municipality_prices is not None else pd.DataFrame()
    mun_boundaries = municipality_boundaries
    if mun_boundaries is None:
        mun_boundaries = {"type": "FeatureCollection", "features": []}

    municipality_prices_bytes = _write_municipality_prices_snapshot(
        mun_prices, data_paths.MUNICIPALITY_PRICES_SNAPSHOT_FILE
    )
    municipality_boundaries_bytes = _write_municipality_boundaries_snapshot(
        mun_boundaries, data_paths.MUNICIPALITY_BOUNDARIES_SNAPSHOT_FILE
    )

    features = boundaries.get("features") or []
    mun_features = mun_boundaries.get("features") or []
    years = sorted({int(y) for y in mun_prices["year"].unique()}) if not mun_prices.empty else []
    manifest = build_manifest(
        prices_rows=len(prices),
        prices_bytes=prices_bytes,
        boundaries_features=len(features),
        boundaries_bytes=boundaries_bytes,
        cpi_rows=len(cpi),
        cpi_bytes=cpi_bytes,
        demographics_rows=len(demographics),
        demographics_bytes=demographics_bytes,
        municipality_prices_rows=len(mun_prices),
        municipality_prices_bytes=municipality_prices_bytes,
        municipality_prices_years=years,
        municipality_boundaries_features=len(mun_features),
        municipality_boundaries_bytes=municipality_boundaries_bytes,
        fetch_date=fetch_date,
    )
    data_paths.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with data_paths.MANIFEST_FILE.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    check_snapshot_size(manifest)
    return manifest


def build_snapshot(*, refresh: bool = True) -> dict[str, Any]:
    """Fetch (when refresh) and write the full data snapshot."""
    if refresh:
        prices = fetch_prices(refresh=True)
        boundaries = fetch_boundaries()
        cpi = fetch_cpi()
        demographics = fetch_demographics()
        municipality_prices = fetch_municipality_prices()
    else:
        prices = load_prices(refresh=False)
        boundaries = load_boundaries(refresh=False)
        cpi = load_cpi(refresh=False)
        demographics = load_demographics(refresh=False)
        municipality_prices = load_municipality_prices(refresh=False)

    municipality_boundaries = build_municipality_boundaries(boundaries)
    municipality_boundaries = apply_municipality_names_to_boundaries(
        municipality_boundaries, municipality_prices
    )
    municipality_join_report(municipality_prices, municipality_boundaries)

    return write_snapshot(
        prices,
        boundaries,
        cpi,
        demographics,
        municipality_prices,
        municipality_boundaries,
    )


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    return f"{num_bytes / 1024:.1f} KB"


def _print_manifest_summary(manifest: dict[str, Any]) -> None:
    files = manifest.get("files") or {}
    print(f"fetch_date: {manifest.get('fetch_date')}")
    for name, entry in files.items():
        if not isinstance(entry, dict):
            continue
        size = _format_size(int(entry.get("bytes", 0)))
        extra = ""
        if "rows" in entry:
            extra = f", rows={entry['rows']}"
        elif "features" in entry:
            extra = f", features={entry['features']}"
        print(f"  {name}: {size}{extra}")
    total = int(manifest.get("total_bytes", total_snapshot_bytes(manifest)))
    print(f"total: {_format_size(total)}")


def main() -> None:
    """CLI entry: build snapshot with live fetch."""
    manifest = build_snapshot(refresh=True)
    _print_manifest_summary(manifest)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
