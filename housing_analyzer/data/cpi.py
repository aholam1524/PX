"""Fetch and parse Statistics Finland consumer price index (PxWeb json-stat2)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from housing_analyzer.data import paths as data_paths
from housing_analyzer.data.prices import PxWebError, _is_missing, _to_float

API_URL = "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/khi/11xs.px"

VAR_TIME = "timeperiod_m"
VAR_CONTENTS = "contentscode"
# Overall index, monthly; 2015=100 (stable base used for real prices).
MEASURE_OVERALL_2015 = "ip_0_2015"
CPI_BASE_YEAR = 2015

_MONTH_RE = re.compile(r"^(\d{4})M(\d{2})$")
_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")

CACHE_DIR = data_paths.RAW_DIR
CACHE_FILE = CACHE_DIR / "cpi.pkl"


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


def _month_to_timestamp(label: str) -> pd.Timestamp:
    match = _MONTH_RE.match(label.strip())
    if not match:
        raise ValueError(f"Invalid CPI month label: {label!r}")
    year = int(match.group(1))
    month = int(match.group(2))
    return pd.Timestamp(year=year, month=month, day=1)


def parse_cpi_json_stat2(dataset: dict[str, Any]) -> pd.DataFrame:
    """Parse CPI json-stat2 into columns ``month`` (month-start) and ``cpi``."""
    if dataset.get("class") != "dataset":
        status = dataset.get("status") or dataset.get("error")
        raise ValueError(f"Not a json-stat2 dataset: {status!r}")

    dim_ids: list[str] = list(dataset["id"])
    sizes: list[int] = list(dataset["size"])
    values: list[Any] = list(dataset.get("value") or [])

    codes_by_dim = {
        dim_id: sorted(
            dataset["dimension"][dim_id]["category"]["index"].keys(),
            key=lambda k: dataset["dimension"][dim_id]["category"]["index"][k],
        )
        for dim_id in dim_ids
    }

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
        if keyed.get(VAR_CONTENTS) != MEASURE_OVERALL_2015:
            continue
        month_label = keyed[VAR_TIME]
        if _is_missing(raw_value):
            cpi = float("nan")
        else:
            cpi = _to_float(raw_value)
        rows.append({"month": _month_to_timestamp(month_label), "cpi": cpi})

    if not rows:
        return pd.DataFrame(columns=["month", "cpi"])

    frame = pd.DataFrame(rows).sort_values("month", ignore_index=True)
    return frame


def _build_cpi_query(months: list[str]) -> dict[str, Any]:
    return {
        "query": [
            {
                "code": VAR_TIME,
                "selection": {"filter": "item", "values": months},
            },
            {
                "code": VAR_CONTENTS,
                "selection": {"filter": "item", "values": [MEASURE_OVERALL_2015]},
            },
        ],
        "response": {"format": "json-stat2"},
    }


def fetch_cpi(
    *,
    api_url: str = API_URL,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    get_metadata: Callable[[str], dict[str, Any]] | None = None,
    pause_seconds: float = 0.0,
) -> pd.DataFrame:
    """Download the overall CPI (2015=100) monthly series from PxWeb."""
    post = post_json or _default_post
    get_meta = get_metadata or _default_get_metadata

    try:
        metadata = get_meta(api_url)
    except requests.RequestException as exc:
        raise PxWebError(f"Could not reach Statistics Finland PxWeb API: {exc}") from exc

    months = [
        value
        for var in metadata.get("variables") or []
        if var.get("code") == VAR_TIME
        for value in var.get("values") or []
    ]
    if not months:
        raise PxWebError("PxWeb CPI metadata missing month codes")

    payload = _build_cpi_query(months)
    try:
        dataset = post(api_url, payload)
    except PxWebError:
        raise
    except Exception as exc:
        raise PxWebError(f"PxWeb CPI request failed: {exc}") from exc

    if pause_seconds:
        time.sleep(pause_seconds)

    return parse_cpi_json_stat2(dataset)


def _quarter_label(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def cpi_by_quarter(cpi_df: pd.DataFrame) -> pd.Series:
    """Quarterly CPI as the mean of three months; quarter labels like ``2025Q4``.

    Quarters with fewer than three observed months in ``cpi_df`` are omitted
    (typically the latest, still-incomplete quarter).
    """
    if cpi_df.empty:
        return pd.Series(dtype=float)

    work = cpi_df.copy()
    work["year"] = work["month"].dt.year
    work["q"] = (work["month"].dt.month - 1) // 3 + 1
    work["quarter"] = work.apply(
        lambda row: _quarter_label(int(row["year"]), int(row["q"])), axis=1
    )

    grouped = work.groupby("quarter", sort=False)
    counts = grouped["cpi"].count()
    means = grouped["cpi"].mean()
    complete = means.loc[counts >= 3].dropna()
    complete = complete.reindex(sorted(complete.index, key=_quarter_sort_key))
    complete.name = "cpi"
    return complete


def _quarter_sort_key(quarter: str) -> int:
    match = _QUARTER_RE.match(quarter)
    if not match:
        return 0
    year = int(match.group(1))
    q = int(match.group(2))
    return year * 4 + (q - 1)


def latest_complete_quarter(cpi_quarterly: pd.Series) -> str | None:
    """Return the latest quarter label in a quarterly CPI series."""
    if cpi_quarterly.empty:
        return None
    ordered = sorted(cpi_quarterly.index, key=_quarter_sort_key)
    return ordered[-1]


def _read_cpi_snapshot(path: Path | None = None) -> pd.DataFrame:
    path = path or data_paths.CPI_SNAPSHOT_FILE
    frame = pd.read_csv(path, compression="gzip", parse_dates=["month"])
    return frame[["month", "cpi"]]


def _load_cpi_fixture() -> pd.DataFrame:
    import json

    path = data_paths.CPI_FIXTURE_FILE
    with path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    return parse_cpi_json_stat2(dataset)


def load_cpi(*, refresh: bool = False) -> pd.DataFrame:
    """Load CPI from snapshot, cache, or PxWeb."""
    if data_paths.use_fixtures():
        return _load_cpi_fixture()

    if refresh:
        frame = fetch_cpi()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(CACHE_FILE)
        return frame

    if data_paths.CPI_SNAPSHOT_FILE.is_file():
        return _read_cpi_snapshot()

    if CACHE_FILE.exists():
        return pd.read_pickle(CACHE_FILE)

    frame = fetch_cpi()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(CACHE_FILE)
    return frame
