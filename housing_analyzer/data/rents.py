"""Fetch and parse Statistics Finland average rent tables (PxWeb json-stat2)."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from housing_analyzer.data import paths as data_paths
from housing_analyzer.data.prices import (
    PxWebError,
    _category_codes,
    _category_labels,
    _is_missing,
    _to_float,
    _to_int,
    read_max_cells,
)

API_URL = "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/asvu/15fa.px"
TABLE_ID = "15fa.px"

CLASSIFICATION_MAPS_URL = (
    "https://data.stat.fi/api/classifications/v2/correspondenceTables/"
    "kunta_1_20250101%23maakunta_1_20250101/maps?content=data&meta=min&lang=en"
)

VAR_TIME = "timeperiod_q"
VAR_FUNDING = "rahoitus_2_20260101"
VAR_ROOMS = "huoneluku_5_20260101"
VAR_AREA = "alue_44_20260101"
VAR_CONTENTS = "contentscode"

MEASURE_RENT = "asvu_keskineliovuokra"
MEASURE_RENT_OBS = "asvu_keskineliovuokra_lkm"
MEASURE_NEW_RENT = "asvu_keskineliovuokra_u"
MEASURE_NEW_RENT_OBS = "asvu_keskineliovuokra_u_lkm"
MEASURES = (
    MEASURE_RENT,
    MEASURE_RENT_OBS,
    MEASURE_NEW_RENT,
    MEASURE_NEW_RENT_OBS,
)

GREATER_HELSINKI_MUNICIPALITIES = frozenset({"091", "049", "092", "235"})

# Former Itä-Uusimaa (MK03) was merged into Uusimaa (MK01) in 2011.
DEFAULT_RENT_REGION_CODES = frozenset(f"MK{i:02d}" for i in range(1, 20) if i != 3)

_CITY_AREA_CODE_RE = re.compile(r"^\d{3}$")
_REGION_AREA_CODE_RE = re.compile(r"^MK\d{2}$")

CACHE_DIR = data_paths.RAW_DIR
CACHE_FILE = CACHE_DIR / "rents.pkl"

logger = logging.getLogger(__name__)


def _normalize_municipality_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text.zfill(3)


def _funding_display(code: str, label: str | None = None) -> str:
    readable = label or code
    return f"{code} — {readable}"


def _rooms_display(code: str, label: str | None = None) -> str:
    readable = label or code
    return f"{code} — {readable}"


def _area_name_from_label(code: str, label: str) -> str:
    text = label.strip()
    if text.startswith("MK") and " " in text:
        return text.split(" ", 1)[1]
    return text


def rent_city_codes_from_metadata(metadata: dict[str, Any]) -> frozenset[str]:
    """Municipality numbers that have their own row in the rent area dimension."""
    for var in metadata.get("variables") or []:
        if var.get("code") != VAR_AREA:
            continue
        return frozenset(
            value
            for value in var.get("values") or []
            if _CITY_AREA_CODE_RE.match(str(value))
        )
    return frozenset()


def rent_region_codes_from_metadata(metadata: dict[str, Any]) -> frozenset[str]:
    for var in metadata.get("variables") or []:
        if var.get("code") != VAR_AREA:
            continue
        return frozenset(
            value
            for value in var.get("values") or []
            if _REGION_AREA_CODE_RE.match(str(value))
        )
    return frozenset()


def _log_rent_city_code_drift(metadata: dict[str, Any]) -> None:
    """Warn when the live rent-area metadata disagrees with the hardcoded city list."""
    live_cities = rent_city_codes_from_metadata(metadata)
    if not live_cities:
        return
    default_cities = _default_rent_city_codes()
    missing = sorted(default_cities - live_cities)
    added = sorted(live_cities - default_cities)
    if missing or added:
        logger.warning(
            "Rent city area codes drifted from the hardcoded default list "
            "(missing from live metadata: %s; new in live metadata: %s)",
            ", ".join(missing) or "none",
            ", ".join(added) or "none",
        )


def _default_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        raise PxWebError(
            f"PxWeb request failed ({response.status_code}): {response.text[:500]}"
        )
    data = response.json()
    if data.get("class") != "dataset":
        raise PxWebError(f"PxWeb returned an error response: {data!r}")
    return data


def _default_get_metadata(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise PxWebError(
            f"PxWeb metadata request failed ({response.status_code}): {response.text[:500]}"
        )
    return response.json()


def _default_get_json(url: str) -> Any:
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise PxWebError(
            f"Classification request failed ({response.status_code}): {response.text[:500]}"
        )
    return response.json()


def _column_order() -> list[str]:
    return [
        "area_code",
        "area_name",
        "funding",
        "rooms",
        "quarter",
        "rent_per_sqm",
        "rent_observations",
        "new_rent_per_sqm",
        "new_rent_observations",
    ]


def _empty_rents_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=_column_order())
    frame["rent_observations"] = frame["rent_observations"].astype("Int64")
    frame["new_rent_observations"] = frame["new_rent_observations"].astype("Int64")
    return frame


def parse_rents_json_stat2(dataset: dict[str, Any]) -> pd.DataFrame:
    """Parse one json-stat2 rent dataset into the tidy rent table."""
    if dataset.get("class") != "dataset":
        status = dataset.get("status") or dataset.get("error")
        raise ValueError(f"Not a json-stat2 dataset: {status!r}")

    dim_ids: list[str] = list(dataset["id"])
    sizes: list[int] = list(dataset["size"])
    values: list[Any] = list(dataset.get("value") or [])

    codes_by_dim = {dim_id: _category_codes(dataset, dim_id) for dim_id in dim_ids}
    labels_by_dim = {dim_id: _category_labels(dataset, dim_id) for dim_id in dim_ids}

    stride = 1
    strides: list[int] = []
    for size in reversed(sizes):
        strides.insert(0, stride)
        stride *= size

    rows: list[dict[str, Any]] = []
    for flat_index, raw_value in enumerate(values):
        remainder = flat_index
        coords: list[int] = []
        for dim_index, dim_size in enumerate(sizes):
            coord = remainder // strides[dim_index]
            remainder = remainder % strides[dim_index]
            coords.append(coord)

        keyed = {
            dim_ids[i]: codes_by_dim[dim_ids[i]][coords[i]] for i in range(len(dim_ids))
        }
        measure = keyed[VAR_CONTENTS]
        if measure not in MEASURES:
            continue

        quarter = keyed[VAR_TIME].rstrip("*")
        area_code = keyed[VAR_AREA]
        area_label = labels_by_dim[VAR_AREA].get(area_code, area_code)
        funding_code = keyed[VAR_FUNDING]
        rooms_code = keyed[VAR_ROOMS]

        row_key = (quarter, area_code, funding_code, rooms_code)
        if not rows or rows[-1].get("_key") != row_key:
            rows.append(
                {
                    "_key": row_key,
                    "area_code": area_code,
                    "area_name": _area_name_from_label(area_code, area_label),
                    "funding": _funding_display(
                        funding_code,
                        labels_by_dim[VAR_FUNDING].get(funding_code),
                    ),
                    "rooms": _rooms_display(
                        rooms_code,
                        labels_by_dim[VAR_ROOMS].get(rooms_code),
                    ),
                    "quarter": quarter,
                    "rent_per_sqm": float("nan"),
                    "rent_observations": pd.NA,
                    "new_rent_per_sqm": float("nan"),
                    "new_rent_observations": pd.NA,
                }
            )
        current = rows[-1]
        if measure == MEASURE_RENT:
            current["rent_per_sqm"] = _to_float(raw_value)
        elif measure == MEASURE_RENT_OBS:
            current["rent_observations"] = _to_int(raw_value)
        elif measure == MEASURE_NEW_RENT:
            current["new_rent_per_sqm"] = _to_float(raw_value)
        elif measure == MEASURE_NEW_RENT_OBS:
            current["new_rent_observations"] = _to_int(raw_value)

    if not rows:
        return _empty_rents_frame()

    frame = pd.DataFrame(rows).drop(columns=["_key"])
    frame["rent_observations"] = frame["rent_observations"].astype("Int64")
    frame["new_rent_observations"] = frame["new_rent_observations"].astype("Int64")
    return frame[_column_order()]


def _build_full_rents_query(
    *,
    quarters: Sequence[str],
    funding: Sequence[str],
    rooms: Sequence[str],
    areas: Sequence[str],
) -> dict[str, Any]:
    return {
        "query": [
            {
                "code": VAR_TIME,
                "selection": {"filter": "item", "values": list(quarters)},
            },
            {
                "code": VAR_FUNDING,
                "selection": {"filter": "item", "values": list(funding)},
            },
            {
                "code": VAR_ROOMS,
                "selection": {"filter": "item", "values": list(rooms)},
            },
            {
                "code": VAR_AREA,
                "selection": {"filter": "item", "values": list(areas)},
            },
            {
                "code": VAR_CONTENTS,
                "selection": {"filter": "item", "values": list(MEASURES)},
            },
        ],
        "response": {"format": "json-stat2"},
    }


def _metadata_variable_values(metadata: dict[str, Any], code: str) -> list[str]:
    for var in metadata.get("variables") or []:
        if var.get("code") == code:
            return list(var.get("values") or [])
    raise PxWebError(f"PxWeb metadata missing variable {code!r}")


def fetch_rents(
    *,
    api_url: str = API_URL,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    get_metadata: Callable[[str], dict[str, Any]] | None = None,
    pause_seconds: float = 0.0,
) -> pd.DataFrame:
    """Download the full quarterly rent table from PxWeb."""
    post = post_json or _default_post
    get_meta = get_metadata or _default_get_metadata

    try:
        metadata = get_meta(api_url)
    except requests.RequestException as exc:
        raise PxWebError(f"Could not reach Statistics Finland PxWeb API: {exc}") from exc

    _log_rent_city_code_drift(metadata)

    max_cells = read_max_cells(metadata)
    quarters = _metadata_variable_values(metadata, VAR_TIME)
    funding = _metadata_variable_values(metadata, VAR_FUNDING)
    rooms = _metadata_variable_values(metadata, VAR_ROOMS)
    areas = _metadata_variable_values(metadata, VAR_AREA)

    n_cells = len(quarters) * len(funding) * len(rooms) * len(areas) * len(MEASURES)
    if n_cells > max_cells:
        raise PxWebError(
            f"Rent table query would exceed PxWeb cell limit ({n_cells} > {max_cells})"
        )

    payload = _build_full_rents_query(
        quarters=quarters,
        funding=funding,
        rooms=rooms,
        areas=areas,
    )
    try:
        dataset = post(api_url, payload)
    except PxWebError:
        raise
    except Exception as exc:
        raise PxWebError(f"PxWeb rent request failed: {exc}") from exc

    if pause_seconds:
        time.sleep(pause_seconds)

    frame = parse_rents_json_stat2(dataset)
    return frame.sort_values(
        ["quarter", "area_code", "funding", "rooms"], ignore_index=True
    )


def _maakunta_to_rent_region(region_code: str, rent_regions: frozenset[str]) -> str | None:
    try:
        rent_region = f"MK{int(region_code):02d}"
    except ValueError:
        return None
    if rent_region in rent_regions:
        return rent_region
    return None


def fetch_municipality_region_map(
    *,
    maps_url: str = CLASSIFICATION_MAPS_URL,
    get_json: Callable[[str], Any] | None = None,
    rent_regions: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Build municipality-to-region correspondence from Statistics Finland classifications."""
    get = get_json or _default_get_json
    try:
        entries = get(maps_url)
    except requests.RequestException as exc:
        raise PxWebError(
            f"Could not reach Statistics Finland classifications API: {exc}"
        ) from exc

    if not isinstance(entries, list):
        raise PxWebError(f"Unexpected classifications response: {entries!r}")

    regions = rent_regions or DEFAULT_RENT_REGION_CODES
    rows: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = entry.get("sourceItem") or {}
        target = entry.get("targetItem") or {}
        municipality_number = _normalize_municipality_number(source.get("code"))
        region_code = str(target.get("code") or "").strip()
        if municipality_number is None or not region_code:
            continue
        names = source.get("classificationItemNames") or []
        municipality_name = next(
            (item.get("name") for item in names if item.get("lang") == "en"),
            names[0].get("name") if names else municipality_number,
        )
        region_names = target.get("classificationItemNames") or []
        region_name = next(
            (item.get("name") for item in region_names if item.get("lang") == "en"),
            region_names[0].get("name") if region_names else region_code,
        )
        rent_region = _maakunta_to_rent_region(region_code, regions)
        rows.append(
            {
                "municipality_number": municipality_number,
                "municipality_name": str(municipality_name),
                "region_code": region_code.zfill(2),
                "region_name": str(region_name),
                "rent_region_code": rent_region or "",
            }
        )

    if not rows:
        raise PxWebError("Classifications API returned no municipality-region rows")

    frame = pd.DataFrame(rows).sort_values("municipality_number", ignore_index=True)
    return frame


def load_municipality_region_map(*, refresh: bool = False) -> pd.DataFrame:
    """Load municipality-region correspondence from snapshot or the classifications API."""
    if data_paths.use_fixtures():
        return _load_municipality_region_fixture()

    if not refresh and data_paths.MUNICIPALITY_REGION_SNAPSHOT_FILE.is_file():
        return pd.read_csv(data_paths.MUNICIPALITY_REGION_SNAPSHOT_FILE, dtype=str)

    frame = fetch_municipality_region_map()
    data_paths.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_paths.MUNICIPALITY_REGION_SNAPSHOT_FILE, index=False)
    return frame


def _load_municipality_region_fixture() -> pd.DataFrame:
    path = data_paths.MUNICIPALITY_REGION_FIXTURE_FILE
    return pd.read_csv(path, dtype=str)


def municipality_numbers_missing_from_region_map(
    municipality_numbers: Sequence[str],
    region_map: pd.DataFrame,
) -> tuple[str, ...]:
    """Return municipality numbers that are absent from the region correspondence."""
    known = {
        _normalize_municipality_number(value)
        for value in region_map["municipality_number"].astype(str)
    }
    known.discard(None)
    missing: list[str] = []
    for raw in municipality_numbers:
        code = _normalize_municipality_number(raw)
        if code is None:
            continue
        if code not in known:
            missing.append(code)
    return tuple(sorted(set(missing)))


def rent_area_for_municipality(
    municipality_number: str | int,
    *,
    region_map: pd.DataFrame | None = None,
    rent_city_codes: frozenset[str] | None = None,
    rent_region_codes: frozenset[str] | None = None,
) -> str:
    """Return the most specific rent area code available for a municipality number."""
    code = _normalize_municipality_number(municipality_number)
    if code is None:
        return "msu"

    cities = rent_city_codes if rent_city_codes is not None else _default_rent_city_codes()
    if code in cities:
        return code

    regions = (
        rent_region_codes
        if rent_region_codes is not None
        else frozenset(f"MK{i:02d}" for i in range(1, 20) if i != 3)
    )

    if region_map is not None and not region_map.empty:
        match = region_map.loc[
            region_map["municipality_number"].astype(str).str.zfill(3) == code
        ]
        if not match.empty:
            rent_region = str(match.iloc[0].get("rent_region_code") or "").strip()
            if rent_region and rent_region in regions:
                return rent_region

    if code in GREATER_HELSINKI_MUNICIPALITIES:
        return "pks"

    return "msu"


def _default_rent_city_codes() -> frozenset[str]:
    return frozenset(
        {
            "091",
            "049",
            "092",
            "638",
            "106",
            "186",
            "245",
            "853",
            "609",
            "684",
            "109",
            "694",
            "837",
            "398",
            "285",
            "286",
            "405",
            "491",
            "297",
            "167",
            "179",
            "743",
            "905",
            "272",
            "564",
            "205",
            "698",
        }
    )


def _read_rents_snapshot(path: Path | None = None) -> pd.DataFrame:
    path = path or data_paths.RENTS_SNAPSHOT_FILE
    frame = pd.read_csv(path, compression="gzip")
    for col in ("rent_observations", "new_rent_observations"):
        if col in frame.columns:
            frame[col] = frame[col].astype("Int64")
    return frame[_column_order()]


def _load_rents_fixture() -> pd.DataFrame:
    import json

    path = data_paths.RENTS_FIXTURE_FILE
    with path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    return parse_rents_json_stat2(dataset)


def load_rents(*, refresh: bool = False) -> pd.DataFrame:
    """Load rents from snapshot, cache, or PxWeb."""
    if data_paths.use_fixtures():
        return _load_rents_fixture()

    if refresh:
        frame = fetch_rents()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(CACHE_FILE)
        return frame

    if data_paths.RENTS_SNAPSHOT_FILE.is_file():
        return _read_rents_snapshot()

    if CACHE_FILE.exists():
        cached = pd.read_pickle(CACHE_FILE)
        return cached[_column_order()]

    frame = fetch_rents()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(CACHE_FILE)
    return frame


def write_municipality_region_snapshot(
    frame: pd.DataFrame,
    path: Path | None = None,
) -> int:
    path = path or data_paths.MUNICIPALITY_REGION_SNAPSHOT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path.stat().st_size
