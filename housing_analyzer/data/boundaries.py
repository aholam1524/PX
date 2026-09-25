"""Fetch and simplify Finnish postal-code area boundaries (Statistics Finland WFS)."""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)

WFS_BASE = "https://geo.stat.fi/geoserver/postialue/wfs"
FEATURE_TYPE = "postialue:pno_2022"
EDITION_LABEL = "pno_2022 (2022 Paavo postal-code areas)"

POSTAL_PROPERTY = "posti_alue"
NAME_PROPERTY = "nimi"

DEFAULT_SIMPLIFY_TOLERANCE = 0.001

from housing_analyzer.data import paths as data_paths

CACHE_DIR = data_paths.RAW_DIR
CACHE_FILE = CACHE_DIR / "housing_boundaries.geojson"
EDITION_FILE = CACHE_DIR / "housing_boundaries_edition.txt"


class WfsError(RuntimeError):
    """Statistics Finland WFS request failed."""


@dataclass(frozen=True)
class JoinReport:
    """Joined price/boundary table plus postal codes only in one source."""

    frame: pd.DataFrame
    only_in_prices: tuple[str, ...]
    only_in_boundaries: tuple[str, ...]


def _normalize_postal_code(value: Any) -> str:
    return str(value).strip().zfill(5)


def _wfs_get_feature_url(type_name: str = FEATURE_TYPE) -> str:
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
    }
    query = "&".join(f"{key}={requests.utils.quote(str(val))}" for key, val in params.items())
    return f"{WFS_BASE}?{query}"


def _default_fetch_geojson(url: str) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=300)
    except requests.RequestException as exc:
        raise WfsError(f"Could not reach Statistics Finland WFS: {exc}") from exc
    if response.status_code != 200:
        raise WfsError(
            f"WFS request failed ({response.status_code}): {response.text[:500]}"
        )
    data = response.json()
    if data.get("type") != "FeatureCollection":
        raise WfsError(f"Unexpected WFS response: {data!r}")
    return data


def simplify_geojson(
    collection: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
) -> dict[str, Any]:
    """Return a GeoJSON FeatureCollection with simplified geometries in EPSG:4326."""
    features_out: list[dict[str, Any]] = []
    for feature in collection.get("features") or []:
        if not isinstance(feature, dict):
            continue
        geom = feature.get("geometry")
        if not geom:
            continue
        try:
            geom_shapely = shape(geom)
        except (TypeError, ValueError):
            continue
        if geom_shapely.is_empty:
            continue
        simplified = geom_shapely.simplify(tolerance, preserve_topology=True)
        props = dict(feature.get("properties") or {})
        postal = _normalize_postal_code(props.get(POSTAL_PROPERTY, ""))
        props[POSTAL_PROPERTY] = postal
        props["postal_code"] = postal
        features_out.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": mapping(simplified),
            }
        )
    return {"type": "FeatureCollection", "features": features_out}


def fetch_boundaries(
    *,
    fetch_geojson: Callable[[str], dict[str, Any]] | None = None,
    type_name: str = FEATURE_TYPE,
) -> dict[str, Any]:
    """Download postal-code boundaries from WFS and return simplified GeoJSON."""
    fetch = fetch_geojson or _default_fetch_geojson
    raw = fetch(_wfs_get_feature_url(type_name))
    return simplify_geojson(raw)


def _read_boundaries_snapshot(path: Path | None = None) -> dict[str, Any]:
    path = path or data_paths.BOUNDARIES_SNAPSHOT_FILE
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot_boundaries_available() -> bool:
    return data_paths.BOUNDARIES_SNAPSHOT_FILE.is_file()


def _load_boundaries_fixture() -> dict[str, Any]:
    with data_paths.BOUNDARIES_FIXTURE_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_boundaries(
    *,
    refresh: bool = False,
    simplify_tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
    fetch_geojson: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load simplified boundaries from snapshot, cache, or WFS."""
    if data_paths.use_fixtures():
        return _load_boundaries_fixture()

    if refresh:
        collection = fetch_boundaries(fetch_geojson=fetch_geojson)
        if simplify_tolerance != DEFAULT_SIMPLIFY_TOLERANCE:
            collection = simplify_geojson(collection, tolerance=simplify_tolerance)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with CACHE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(collection, handle, ensure_ascii=False)
        EDITION_FILE.write_text(f"{FEATURE_TYPE}\n{EDITION_LABEL}\n", encoding="utf-8")
        return collection

    if _snapshot_boundaries_available():
        return _read_boundaries_snapshot()

    if CACHE_FILE.exists() and EDITION_FILE.exists():
        with CACHE_FILE.open(encoding="utf-8") as handle:
            return json.load(handle)

    collection = fetch_boundaries(fetch_geojson=fetch_geojson)
    if simplify_tolerance != DEFAULT_SIMPLIFY_TOLERANCE:
        collection = simplify_geojson(collection, tolerance=simplify_tolerance)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(collection, handle, ensure_ascii=False)
    EDITION_FILE.write_text(f"{FEATURE_TYPE}\n{EDITION_LABEL}\n", encoding="utf-8")
    return collection


def _boundary_index(
    boundaries: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_code: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    for feature in boundaries.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        code = _normalize_postal_code(props.get("postal_code") or props.get(POSTAL_PROPERTY))
        by_code[code] = feature
        name = props.get(NAME_PROPERTY)
        if isinstance(name, str) and name.strip():
            names[code] = name.strip()
    return by_code, names


def _prices_for_quarter(
    prices_df: pd.DataFrame,
    quarter: str,
    building_type: str | None,
) -> pd.DataFrame:
    frame = prices_df.loc[prices_df["quarter"] == quarter].copy()
    if building_type is not None:
        bt = str(building_type)
        frame = frame.loc[
            frame["building_type"].eq(bt)
            | frame["building_type"].str.startswith(f"{bt} —")
            | frame["building_type"].str.startswith(f"{bt} -")
        ]
    if frame.empty:
        return frame
    frame["postal_code"] = frame["postal_code"].astype(str).str.zfill(5)
    frame = frame.sort_values(["postal_code", "building_type"], kind="stable")
    return frame.drop_duplicates(subset=["postal_code"], keep="first")


def join_prices_to_areas(
    prices_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    quarter: str,
    building_type: str | None = None,
) -> JoinReport:
    """Join one quarter of prices to boundary areas; report unmatched postal codes."""
    price_slice = _prices_for_quarter(prices_df, quarter, building_type)
    boundary_features, boundary_names = _boundary_index(boundaries)

    price_codes = set(price_slice["postal_code"].tolist()) if not price_slice.empty else set()
    boundary_codes = set(boundary_features.keys())

    only_in_prices = tuple(sorted(price_codes - boundary_codes))
    only_in_boundaries = tuple(sorted(boundary_codes - price_codes))

    if only_in_prices or only_in_boundaries:
        logger.info(
            "Postal codes only in prices (%d): %s",
            len(only_in_prices),
            ", ".join(only_in_prices[:20]) + ("…" if len(only_in_prices) > 20 else ""),
        )
        logger.info(
            "Postal codes only in boundaries (%d): %s",
            len(only_in_boundaries),
            ", ".join(only_in_boundaries[:20])
            + ("…" if len(only_in_boundaries) > 20 else ""),
        )

    all_codes = sorted(price_codes | boundary_codes)
    rows: list[dict[str, Any]] = []
    for code in all_codes:
        in_prices = code in price_codes
        in_boundaries = code in boundary_codes
        price_row = (
            price_slice.loc[price_slice["postal_code"] == code].iloc[0]
            if in_prices
            else None
        )
        area_name = ""
        if price_row is not None:
            name_val = price_row["area_name"]
            if isinstance(name_val, str):
                area_name = name_val
        elif in_boundaries:
            area_name = boundary_names.get(code, "")

        rows.append(
            {
                "postal_code": code,
                "area_name": area_name,
                "price_per_sqm": price_row["price_per_sqm"] if price_row is not None else float("nan"),
                "transactions": price_row["transactions"] if price_row is not None else pd.NA,
                "has_boundary": in_boundaries,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["transactions"] = frame["transactions"].astype("Int64")
    return JoinReport(
        frame=frame,
        only_in_prices=only_in_prices,
        only_in_boundaries=only_in_boundaries,
    )
