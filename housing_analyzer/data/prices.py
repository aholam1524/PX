"""Fetch and parse Statistics Finland housing price tables (PxWeb json-stat2)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_URL = (
    "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/ashi/13mt.px"
)
TABLE_TITLE = (
    "Prices per square meter of old dwellings in housing companies and "
    "numbers of transactions by postal code area, quarterly"
)

VAR_TIME = "timeperiod_q"
VAR_POSTAL = "postinumeroalue_4_20220101"
VAR_BUILDING = "talotyyppi_6_20131021"
VAR_CONTENTS = "contentscode"

MEASURE_PRICE = "keskihinta_aritm_nw"
MEASURE_TRANSACTIONS = "lkm_julk20"
MEASURES = (MEASURE_PRICE, MEASURE_TRANSACTIONS)

# PxWeb allows at most 100_000 cells per query; stay below that.
DEFAULT_MAX_CELLS = 100_000
SAFE_CELL_MARGIN = 0.9

BUILDING_TYPE_LABELS = {
    "1": "Blocks of flats, one-room flat",
    "2": "Blocks of flats, two-room flat",
    "3": "Blocks of flats, three-room flat+",
    "5": "Terraced houses total",
}

_MISSING_STRINGS = frozenset({"..", "-", ""})

_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO_ROOT / "data" / "raw"
CACHE_FILE = CACHE_DIR / "housing_prices.pkl"

_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])")


class PxWebError(RuntimeError):
    """Statistics Finland PxWeb request failed."""


def _quarter_period_end(quarter: str) -> date:
    match = _QUARTER_RE.match(quarter.strip().rstrip("*"))
    if not match:
        raise ValueError(f"Invalid quarter label: {quarter!r}")
    year = int(match.group(1))
    q = int(match.group(2))
    ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    month, day = ends[q]
    return date(year, month, day)


def _split_postal_label(code: str, label: str) -> tuple[str, str]:
    postal_code = str(code).zfill(5)
    text = label.strip()
    if text.startswith(postal_code):
        text = text[len(postal_code) :].strip()
    area_name = re.sub(r"\s+", " ", text)
    return postal_code, area_name


def _building_type_display(code: str, label: str | None = None) -> str:
    readable = label or BUILDING_TYPE_LABELS.get(code, code)
    return f"{code} — {readable}"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() in _MISSING_STRINGS
    return False


def _to_float(value: Any) -> float:
    if _is_missing(value):
        return float("nan")
    return float(value)


def _to_int(value: Any) -> Any:
    if _is_missing(value):
        return pd.NA
    return int(float(value))


def read_max_cells(metadata: dict[str, Any]) -> int:
    """Return PxWeb cell limit from metadata when present, else the known default."""
    for key in ("maxCells", "maxcells", "maxNumberOfCells"):
        if key in metadata:
            return int(metadata[key])
    return DEFAULT_MAX_CELLS


def _metadata_variables(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    variables = metadata.get("variables")
    if not isinstance(variables, list):
        raise PxWebError("Unexpected PxWeb metadata: missing variables list")
    return variables


def _variable_by_code(metadata: dict[str, Any], code: str) -> dict[str, Any]:
    for var in _metadata_variables(metadata):
        if var.get("code") == code:
            return var
    raise PxWebError(f"PxWeb metadata missing variable {code!r}")


def plan_fetch_queries(
    metadata: dict[str, Any],
    *,
    max_cells: int | None = None,
) -> list[dict[str, Any]]:
    """Split a full-table request into PxWeb queries that respect the cell limit."""
    limit = max_cells if max_cells is not None else read_max_cells(metadata)
    budget = max(1, int(limit * SAFE_CELL_MARGIN))

    quarters = list(_variable_by_code(metadata, VAR_TIME)["values"])
    postals = list(_variable_by_code(metadata, VAR_POSTAL)["values"])
    building_types = list(_variable_by_code(metadata, VAR_BUILDING)["values"])
    n_measures = len(MEASURES)

    queries: list[dict[str, Any]] = []
    for building in building_types:
        quarter_batches = _split_for_cell_budget(
            quarters, len(postals), 1, n_measures, budget
        )
        for quarter_batch in quarter_batches:
            postal_batches = _split_for_cell_budget(
                postals, len(quarter_batch), 1, n_measures, budget
            )
            for postal_batch in postal_batches:
                queries.append(
                    {
                        "quarters": quarter_batch,
                        "postal_codes": postal_batch,
                        "building_types": [building],
                    }
                )
    return queries


def _split_for_cell_budget(
    items: Sequence[str],
    other_dim_size: int,
    building_types: int,
    n_measures: int,
    budget: int,
) -> list[list[str]]:
    if not items:
        return [[]]
    per_item = other_dim_size * building_types * n_measures
    if per_item <= 0:
        per_item = 1
    chunk_size = max(1, budget // per_item)
    if chunk_size >= len(items):
        return [list(items)]
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def _category_codes(dataset: dict[str, Any], dim_id: str) -> list[str]:
    dim = dataset["dimension"][dim_id]
    index_map = dim["category"]["index"]
    return sorted(index_map.keys(), key=lambda k: index_map[k])


def _category_labels(dataset: dict[str, Any], dim_id: str) -> dict[str, str]:
    return dict(dataset["dimension"][dim_id]["category"]["label"])


def parse_json_stat2(dataset: dict[str, Any]) -> pd.DataFrame:
    """Parse one json-stat2 dataset into the tidy price table."""
    if dataset.get("class") != "dataset":
        status = dataset.get("status") or dataset.get("error")
        raise ValueError(f"Not a json-stat2 dataset: {status!r}")

    dim_ids: list[str] = list(dataset["id"])
    sizes: list[int] = list(dataset["size"])
    values: list[Any] = list(dataset.get("value") or [])

    codes_by_dim = {dim_id: _category_codes(dataset, dim_id) for dim_id in dim_ids}
    labels_by_dim = {dim_id: _category_labels(dataset, dim_id) for dim_id in dim_ids}

    stride = 1
    strides = []
    for size in reversed(sizes):
        strides.insert(0, stride)
        stride *= size

    rows: list[dict[str, Any]] = []
    for flat_index, raw_value in enumerate(values):
        coords = []
        remainder = flat_index
        for dim_index, dim_size in enumerate(sizes):
            strides_for_dim = strides[dim_index]
            coord = remainder // strides_for_dim
            remainder = remainder % strides_for_dim
            coords.append(coord)

        keyed = {
            dim_ids[i]: codes_by_dim[dim_ids[i]][coords[i]] for i in range(len(dim_ids))
        }

        quarter = keyed[VAR_TIME].rstrip("*")
        postal_code, area_name = _split_postal_label(
            keyed[VAR_POSTAL],
            labels_by_dim[VAR_POSTAL][keyed[VAR_POSTAL]],
        )
        building_code = keyed[VAR_BUILDING]
        building_label = labels_by_dim[VAR_BUILDING].get(building_code)
        measure = keyed[VAR_CONTENTS]

        row_key = (quarter, postal_code, building_code)
        if not rows or rows[-1].get("_key") != row_key:
            rows.append(
                {
                    "_key": row_key,
                    "postal_code": postal_code,
                    "area_name": area_name,
                    "building_type": _building_type_display(building_code, building_label),
                    "quarter": quarter,
                    "period_end": _quarter_period_end(quarter),
                    "price_per_sqm": float("nan"),
                    "transactions": pd.NA,
                }
            )
        current = rows[-1]
        if measure == MEASURE_PRICE:
            current["price_per_sqm"] = _to_float(raw_value)
        elif measure == MEASURE_TRANSACTIONS:
            current["transactions"] = _to_int(raw_value)

    if not rows:
        return _empty_prices_frame()

    frame = pd.DataFrame(rows).drop(columns=["_key"])
    frame["postal_code"] = frame["postal_code"].astype(str).str.zfill(5)
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["transactions"] = frame["transactions"].astype("Int64")
    return frame[_column_order()]


def _column_order() -> list[str]:
    return [
        "postal_code",
        "area_name",
        "building_type",
        "quarter",
        "period_end",
        "price_per_sqm",
        "transactions",
    ]


def _empty_prices_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=_column_order())
    frame["transactions"] = frame["transactions"].astype("Int64")
    return frame


def _build_px_query(
    quarters: Sequence[str],
    postal_codes: Sequence[str],
    building_types: Sequence[str],
) -> dict[str, Any]:
    return {
        "query": [
            {
                "code": VAR_TIME,
                "selection": {"filter": "item", "values": list(quarters)},
            },
            {
                "code": VAR_POSTAL,
                "selection": {"filter": "item", "values": list(postal_codes)},
            },
            {
                "code": VAR_BUILDING,
                "selection": {"filter": "item", "values": list(building_types)},
            },
            {
                "code": VAR_CONTENTS,
                "selection": {"filter": "item", "values": list(MEASURES)},
            },
        ],
        "response": {"format": "json-stat2"},
    }


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


def fetch_prices(
    *,
    refresh: bool = False,
    pause_seconds: float = 0.25,
    api_url: str = API_URL,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    get_metadata: Callable[[str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Query PxWeb in chunks and return the combined tidy table."""
    del refresh  # cache is handled by load_prices()

    post = post_json or _default_post
    get_meta = get_metadata or (lambda url: requests.get(url, timeout=60).json())

    try:
        metadata = get_meta(api_url)
    except requests.RequestException as exc:
        raise PxWebError(f"Could not reach Statistics Finland PxWeb API: {exc}") from exc

    max_cells = read_max_cells(metadata)
    chunks = plan_fetch_queries(metadata, max_cells=max_cells)

    frames: list[pd.DataFrame] = []
    for index, chunk in enumerate(chunks):
        payload = _build_px_query(
            chunk["quarters"],
            chunk["postal_codes"],
            chunk["building_types"],
        )
        try:
            dataset = post(api_url, payload)
        except PxWebError:
            raise
        except Exception as exc:
            raise PxWebError(f"PxWeb request failed: {exc}") from exc

        frames.append(parse_json_stat2(dataset))
        if pause_seconds and index + 1 < len(chunks):
            time.sleep(pause_seconds)

    if not frames:
        return _empty_prices_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["postal_code", "building_type", "quarter"], keep="last"
    )
    return combined.sort_values(
        ["quarter", "postal_code", "building_type"], ignore_index=True
    )


def load_prices(*, refresh: bool = False) -> pd.DataFrame:
    """Load housing prices from cache, fetching from PxWeb when needed."""
    if CACHE_FILE.exists() and not refresh:
        return pd.read_pickle(CACHE_FILE)

    frame = fetch_prices()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(CACHE_FILE)
    return frame
