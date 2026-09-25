"""Fetch and parse Statistics Finland Paavo open data by postal code (PxWeb)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from housing_analyzer.data import paths as data_paths
from housing_analyzer.data.prices import PxWebError, _is_missing, _to_float, read_max_cells

PAAVO_BASE = (
    "https://pxdata.stat.fi/PxWeb/api/v1/en/Postinumeroalueittainen_avoin_tieto/uusin"
)
TABLE_POPULATION = f"{PAAVO_BASE}/12ey.px"
TABLE_INCOME = f"{PAAVO_BASE}/12f1.px"
TABLE_EDUCATION = f"{PAAVO_BASE}/12ez.px"

VAR_POSTAL = "postinumeroalue_4_20260101"
VAR_YEAR = "timeperiod_y"
VAR_CONTENTS = "contentscode"

MEASURE_POPULATION = "he_vakiy"
MEASURES_AGE_65_PLUS = ("he_65_69", "he_70_74", "he_75_79", "he_80_84", "he_85_")
MEASURE_MEDIAN_INCOME = "hr_mtu"
MEASURE_AGE_18_PLUS = "ko_ika18y"
MEASURE_HIGHER_EDUCATION = "ko_yl_kork"

# Documented in the feature PR: Paavo tables 12ey / 12f1 / 12ez, year 2024.
DEFAULT_DATA_YEAR = "2024"

_POSTAL_RE = re.compile(r"^\d{5}$")

CACHE_DIR = data_paths.RAW_DIR
CACHE_FILE = CACHE_DIR / "demographics.pkl"

DEMOGRAPHICS_COLUMNS = (
    "postal_code",
    "data_year",
    "population",
    "median_income_eur",
    "share_age_65_plus",
    "share_higher_education",
)


@dataclass(frozen=True)
class DemographicsJoinReport:
    frame: pd.DataFrame
    only_in_prices: tuple[str, ...]
    only_in_demographics: tuple[str, ...]


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


def _variable_values(metadata: dict[str, Any], code: str) -> list[str]:
    for var in metadata.get("variables") or []:
        if var.get("code") == code:
            return list(var.get("values") or [])
    raise PxWebError(f"PxWeb metadata missing variable {code!r}")


def _normalize_postal_code(raw: str) -> str | None:
    text = str(raw).strip()
    if not _POSTAL_RE.match(text):
        return None
    return text.zfill(5)


def _build_query(
    year: str,
    postal_codes: Sequence[str],
    measures: Sequence[str],
) -> dict[str, Any]:
    return {
        "query": [
            {
                "code": VAR_YEAR,
                "selection": {"filter": "item", "values": [year]},
            },
            {
                "code": VAR_POSTAL,
                "selection": {"filter": "item", "values": list(postal_codes)},
            },
            {
                "code": VAR_CONTENTS,
                "selection": {"filter": "item", "values": list(measures)},
            },
        ],
        "response": {"format": "json-stat2"},
    }


def _category_codes(dataset: dict[str, Any], dim_id: str) -> list[str]:
    dim = dataset["dimension"][dim_id]
    index_map = dim["category"]["index"]
    return sorted(index_map.keys(), key=lambda k: index_map[k])


def parse_paavo_long(dataset: dict[str, Any]) -> pd.DataFrame:
    """Parse json-stat2 into long form: postal_code, measure, value."""
    if dataset.get("class") != "dataset":
        status = dataset.get("status") or dataset.get("error")
        raise ValueError(f"Not a json-stat2 dataset: {status!r}")

    dim_ids: list[str] = list(dataset["id"])
    sizes: list[int] = list(dataset["size"])
    values: list[Any] = list(dataset.get("value") or [])

    codes_by_dim = {dim_id: _category_codes(dataset, dim_id) for dim_id in dim_ids}

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
        postal = _normalize_postal_code(keyed.get(VAR_POSTAL, ""))
        if postal is None:
            continue
        measure = keyed.get(VAR_CONTENTS)
        if measure is None:
            continue
        if _is_missing(raw_value):
            value = float("nan")
        else:
            value = _to_float(raw_value)
        rows.append({"postal_code": postal, "measure": measure, "value": value})

    if not rows:
        return pd.DataFrame(columns=["postal_code", "measure", "value"])
    return pd.DataFrame(rows)


def _chunk_postal_codes(postal_codes: Sequence[str], n_measures: int, budget: int) -> list[list[str]]:
    per_postal = max(1, n_measures)
    chunk_size = max(1, budget // per_postal)
    items = list(postal_codes)
    if chunk_size >= len(items):
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _fetch_table_measures(
    api_url: str,
    year: str,
    measures: Sequence[str],
    *,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]],
    get_metadata: Callable[[str], dict[str, Any]],
    pause_seconds: float = 0.25,
) -> pd.DataFrame:
    metadata = get_metadata(api_url)
    postals = [
        code
        for code in _variable_values(metadata, VAR_POSTAL)
        if _normalize_postal_code(code) is not None
    ]
    budget = int(read_max_cells(metadata) * 0.9)
    chunks = _chunk_postal_codes(postals, len(measures), budget)

    frames: list[pd.DataFrame] = []
    for index, postal_batch in enumerate(chunks):
        payload = _build_query(year, postal_batch, measures)
        dataset = post_json(api_url, payload)
        frames.append(parse_paavo_long(dataset))
        if pause_seconds and index + 1 < len(chunks):
            time.sleep(pause_seconds)

    if not frames:
        return pd.DataFrame(columns=["postal_code", "measure", "value"])
    return pd.concat(frames, ignore_index=True)


def _pivot_measures(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=["postal_code", "measure", "value"])
    return long.pivot_table(
        index="postal_code", columns="measure", values="value", aggfunc="first"
    )


def _assemble_demographics(
    population: pd.DataFrame,
    income: pd.DataFrame,
    education: pd.DataFrame,
    *,
    data_year: str,
) -> pd.DataFrame:
    pop = _pivot_measures(population)
    inc = _pivot_measures(income)
    edu = _pivot_measures(education)

    def _measure_value(pivot: pd.DataFrame, measure: str, code: str) -> float:
        if pivot.empty or measure not in pivot.columns or code not in pivot.index:
            return float("nan")
        return float(pivot.at[code, measure])

    codes = sorted(set(pop.index) | set(inc.index) | set(edu.index))
    rows: list[dict[str, Any]] = []
    for code in codes:
        pop_total = _measure_value(pop, MEASURE_POPULATION, code)
        age_counts: list[float] = []
        for measure in MEASURES_AGE_65_PLUS:
            val = _measure_value(pop, measure, code)
            if pd.notna(val):
                age_counts.append(float(val))
        age_65_plus = sum(age_counts) if age_counts else float("nan")

        share_65 = float("nan")
        if pd.notna(pop_total) and pop_total > 0 and age_counts:
            share_65 = age_65_plus / pop_total

        median_income = _measure_value(inc, MEASURE_MEDIAN_INCOME, code)
        adults = _measure_value(edu, MEASURE_AGE_18_PLUS, code)
        higher = _measure_value(edu, MEASURE_HIGHER_EDUCATION, code)
        share_higher = float("nan")
        if pd.notna(adults) and adults > 0 and pd.notna(higher):
            share_higher = higher / adults

        rows.append(
            {
                "postal_code": str(code).zfill(5),
                "data_year": int(data_year),
                "population": pop_total if pd.notna(pop_total) else float("nan"),
                "median_income_eur": median_income,
                "share_age_65_plus": share_65,
                "share_higher_education": share_higher,
            }
        )

    frame = pd.DataFrame(rows)
    return frame[list(DEMOGRAPHICS_COLUMNS)]


def fetch_demographics(
    *,
    data_year: str = DEFAULT_DATA_YEAR,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    get_metadata: Callable[[str], dict[str, Any]] | None = None,
    pause_seconds: float = 0.25,
) -> pd.DataFrame:
    """Download Paavo demographics for one reference year."""
    post = post_json or _default_post
    get_meta = get_metadata or _default_get_metadata

    try:
        pop_long = _fetch_table_measures(
            TABLE_POPULATION,
            data_year,
            [MEASURE_POPULATION, *MEASURES_AGE_65_PLUS],
            post_json=post,
            get_metadata=get_meta,
            pause_seconds=pause_seconds,
        )
        income_long = _fetch_table_measures(
            TABLE_INCOME,
            data_year,
            [MEASURE_MEDIAN_INCOME],
            post_json=post,
            get_metadata=get_meta,
            pause_seconds=pause_seconds,
        )
        edu_long = _fetch_table_measures(
            TABLE_EDUCATION,
            data_year,
            [MEASURE_AGE_18_PLUS, MEASURE_HIGHER_EDUCATION],
            post_json=post,
            get_metadata=get_meta,
            pause_seconds=pause_seconds,
        )
    except requests.RequestException as exc:
        raise PxWebError(f"Could not reach Statistics Finland PxWeb API: {exc}") from exc

    return _assemble_demographics(pop_long, income_long, edu_long, data_year=data_year)


def join_demographics_to_postal_codes(
    demographics: pd.DataFrame,
    postal_codes: Sequence[str],
) -> DemographicsJoinReport:
    """Left-join demographics onto postal codes; report codes only in one source."""
    demo = demographics.copy()
    demo["postal_code"] = demo["postal_code"].astype(str).str.zfill(5)
    demo_by_code = demo.set_index("postal_code")

    normalized = [str(code).zfill(5) for code in postal_codes]
    price_codes = set(normalized)
    demo_codes = set(demo_by_code.index.astype(str))

    only_in_prices = tuple(sorted(price_codes - demo_codes))
    only_in_demographics = tuple(sorted(demo_codes - price_codes))

    rows: list[dict[str, Any]] = []
    for code in sorted(price_codes):
        row = {"postal_code": code}
        if code in demo_by_code.index:
            for col in DEMOGRAPHICS_COLUMNS:
                if col == "postal_code":
                    continue
                row[col] = demo_by_code.loc[code, col]
        else:
            for col in DEMOGRAPHICS_COLUMNS:
                if col == "postal_code":
                    continue
                row[col] = float("nan") if col != "data_year" else pd.NA
        rows.append(row)

    return DemographicsJoinReport(
        frame=pd.DataFrame(rows),
        only_in_prices=only_in_prices,
        only_in_demographics=only_in_demographics,
    )


def _read_demographics_snapshot(path: Path | None = None) -> pd.DataFrame:
    path = path or data_paths.DEMOGRAPHICS_SNAPSHOT_FILE
    frame = pd.read_csv(path, compression="gzip")
    frame["postal_code"] = frame["postal_code"].astype(str).str.zfill(5)
    if "data_year" in frame.columns:
        frame["data_year"] = frame["data_year"].astype("Int64")
    return frame[list(DEMOGRAPHICS_COLUMNS)]


def _snapshot_demographics_available() -> bool:
    return data_paths.DEMOGRAPHICS_SNAPSHOT_FILE.is_file()


def _load_demographics_fixture() -> pd.DataFrame:
    path = data_paths.DEMOGRAPHICS_FIXTURE_FILE
    frame = pd.read_csv(path)
    frame["postal_code"] = frame["postal_code"].astype(str).str.zfill(5)
    return frame[list(DEMOGRAPHICS_COLUMNS)]


def attach_demographics_to_summaries(
    summaries: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Add median income (and other demo columns) aligned to summary index."""
    out = summaries.copy()
    demo = demographics.copy()
    demo["postal_code"] = demo["postal_code"].astype(str).str.zfill(5)
    demo_index = demo.set_index("postal_code")
    index_codes = out.index.astype(str).str.zfill(5)
    for col in (
        "median_income_eur",
        "population",
        "share_age_65_plus",
        "share_higher_education",
        "data_year",
    ):
        if col in demo_index.columns:
            out[col] = demo_index.reindex(index_codes)[col].to_numpy()
    return out


def load_demographics(*, refresh: bool = False) -> pd.DataFrame:
    """Load demographics from snapshot, cache, or PxWeb."""
    if data_paths.use_fixtures():
        return _load_demographics_fixture()

    if refresh:
        frame = fetch_demographics()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(CACHE_FILE)
        return frame

    if _snapshot_demographics_available():
        return _read_demographics_snapshot()

    if CACHE_FILE.exists():
        frame = pd.read_pickle(CACHE_FILE)
        frame["postal_code"] = frame["postal_code"].astype(str).str.zfill(5)
        return frame

    frame = fetch_demographics()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(CACHE_FILE)
    return frame
