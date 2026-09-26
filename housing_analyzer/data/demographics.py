"""Fetch and parse Statistics Finland Paavo open data by postal code (PxWeb)."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
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
TABLE_HOUSEHOLDS = f"{PAAVO_BASE}/12f2.px"
TABLE_HOUSEHOLD_INCOME = f"{PAAVO_BASE}/12f3.px"
TABLE_DWELLINGS = f"{PAAVO_BASE}/12f4.px"
TABLE_ACTIVITY = f"{PAAVO_BASE}/12f6.px"

VAR_POSTAL = "postinumeroalue_4_20260101"
VAR_YEAR = "timeperiod_y"
VAR_CONTENTS = "contentscode"
NATIONAL_POSTAL_CODE = "SSS"

MEASURE_POPULATION = "he_vakiy"
MEASURES_AGE_65_PLUS = ("he_65_69", "he_70_74", "he_75_79", "he_80_84", "he_85_")
MEASURE_MEDIAN_INCOME = "hr_mtu"
MEASURE_AGE_18_PLUS = "ko_ika18y"
MEASURE_HIGHER_EDUCATION = "ko_yl_kork"

MEASURE_HOUSEHOLDS_TOTAL = "te_taly"
MEASURE_AVG_HOUSEHOLD_SIZE = "te_takk"
MEASURE_AVG_FLOOR_AREA_PER_PERSON = "te_as_valj"
MEASURE_HOUSEHOLDS_OWNER = "te_omis_as"
MEASURE_HOUSEHOLDS_RENTED = "te_vuok_as"
# 12f3: median disposable monetary income of households (EUR/year).
MEASURE_MEDIAN_HOUSEHOLD_INCOME = "tr_mtu"

MEASURE_DWELLINGS = "ra_asunn"
MEASURE_AVG_FLOOR_AREA_DWELLING = "ra_as_kpa"
MEASURE_DWELLINGS_BLOCKS = "ra_kt_as"
MEASURE_DWELLINGS_SMALL_HOUSES = "ra_pt_as"

MEASURE_ACTIVITY_INHABITANTS = "pt_vakiy"
MEASURE_EMPLOYED = "pt_tyoll"
MEASURE_UNEMPLOYED = "pt_tyott"
MEASURE_STUDENTS = "pt_opisk"
MEASURE_PENSIONERS = "pt_elakel"

# Documented in the feature PR: Paavo tables 12ey–12f6, year 2024.
DEFAULT_DATA_YEAR = "2024"

_POSTAL_RE = re.compile(r"^\d{5}$")

CACHE_DIR = data_paths.RAW_DIR
CACHE_FILE = CACHE_DIR / "demographics.pkl"
NATIONAL_CACHE_KEY = "national"

DEMOGRAPHICS_COLUMNS = (
    "postal_code",
    "data_year",
    "population",
    "median_income_eur",
    "share_age_65_plus",
    "share_higher_education",
    "households_total",
    "average_household_size",
    "average_floor_area_per_person",
    "households_owner_occupied",
    "households_rented",
    "median_household_income_eur",
    "dwellings",
    "average_floor_area_per_dwelling",
    "dwellings_blocks_of_flats",
    "dwellings_small_houses",
    "activity_inhabitants",
    "employed",
    "unemployed",
    "students",
    "pensioners",
    "rented_share",
    "owner_occupied_share",
    "unemployment_rate",
    "student_share",
    "pensioner_share",
    "share_blocks_of_flats",
)

NATIONAL_DEMOGRAPHICS_COLUMNS = tuple(
    col for col in DEMOGRAPHICS_COLUMNS if col != "postal_code"
)


@dataclass(frozen=True)
class DemographicsBundle:
    areas: pd.DataFrame
    national: pd.Series


@dataclass(frozen=True)
class DemographicsJoinReport:
    frame: pd.DataFrame
    only_in_prices: tuple[str, ...]
    only_in_demographics: tuple[str, ...]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator):
        return float("nan")
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def compute_rented_share(rented: float, households_total: float) -> float:
    return _safe_ratio(rented, households_total)


def compute_owner_occupied_share(owner_occupied: float, households_total: float) -> float:
    return _safe_ratio(owner_occupied, households_total)


def compute_unemployment_rate(unemployed: float, employed: float) -> float:
    """Unemployed share of the labour force (employed + unemployed), not of all inhabitants."""
    labour_force = float("nan")
    if pd.notna(unemployed) and pd.notna(employed):
        labour_force = float(unemployed) + float(employed)
    return _safe_ratio(unemployed, labour_force)


def compute_inhabitant_share(count: float, inhabitants: float) -> float:
    return _safe_ratio(count, inhabitants)


def compute_share_blocks_of_flats(blocks: float, dwellings: float) -> float:
    return _safe_ratio(blocks, dwellings)


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


def _normalize_postal_or_sss(raw: str) -> str | None:
    text = str(raw).strip()
    if text == NATIONAL_POSTAL_CODE:
        return NATIONAL_POSTAL_CODE
    return _normalize_postal_code(text)


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


def parse_paavo_long(
    dataset: dict[str, Any],
    *,
    include_sss: bool = False,
) -> pd.DataFrame:
    """Parse json-stat2 into long form: postal_code, measure, value."""
    if dataset.get("class") != "dataset":
        status = dataset.get("status") or dataset.get("error")
        raise ValueError(f"Not a json-stat2 dataset: {status!r}")

    normalizer = _normalize_postal_or_sss if include_sss else _normalize_postal_code

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
        postal = normalizer(keyed.get(VAR_POSTAL, ""))
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
    postal_codes: Sequence[str] | None = None,
    include_sss: bool = False,
) -> pd.DataFrame:
    if postal_codes is None:
        metadata = get_metadata(api_url)
        postals = [
            code
            for code in _variable_values(metadata, VAR_POSTAL)
            if _normalize_postal_or_sss(code) is not None
            and (include_sss or _normalize_postal_code(code) is not None)
        ]
    else:
        postals = list(postal_codes)

    if not postals:
        return pd.DataFrame(columns=["postal_code", "measure", "value"])

    if len(postals) <= 2:
        payload = _build_query(year, postals, measures)
        dataset = post_json(api_url, payload)
        return parse_paavo_long(dataset, include_sss=include_sss)

    metadata = get_metadata(api_url)
    budget = int(read_max_cells(metadata) * 0.9)
    chunks = _chunk_postal_codes(postals, len(measures), budget)

    frames: list[pd.DataFrame] = []
    for index, postal_batch in enumerate(chunks):
        payload = _build_query(year, postal_batch, measures)
        dataset = post_json(api_url, payload)
        frames.append(parse_paavo_long(dataset, include_sss=include_sss))
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


def _measure_value(pivot: pd.DataFrame, measure: str, code: str) -> float:
    if pivot.empty or measure not in pivot.columns or code not in pivot.index:
        return float("nan")
    return float(pivot.at[code, measure])


def _derived_measures(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "rented_share": compute_rented_share(
            row.get("households_rented", float("nan")),
            row.get("households_total", float("nan")),
        ),
        "owner_occupied_share": compute_owner_occupied_share(
            row.get("households_owner_occupied", float("nan")),
            row.get("households_total", float("nan")),
        ),
        "unemployment_rate": compute_unemployment_rate(
            row.get("unemployed", float("nan")),
            row.get("employed", float("nan")),
        ),
        "student_share": compute_inhabitant_share(
            row.get("students", float("nan")),
            row.get("activity_inhabitants", float("nan")),
        ),
        "pensioner_share": compute_inhabitant_share(
            row.get("pensioners", float("nan")),
            row.get("activity_inhabitants", float("nan")),
        ),
        "share_blocks_of_flats": compute_share_blocks_of_flats(
            row.get("dwellings_blocks_of_flats", float("nan")),
            row.get("dwellings", float("nan")),
        ),
    }


def _row_for_postal_code(
    code: str,
    *,
    data_year: str,
    pop: pd.DataFrame,
    inc: pd.DataFrame,
    edu: pd.DataFrame,
    households: pd.DataFrame,
    household_income: pd.DataFrame,
    dwellings: pd.DataFrame,
    activity: pd.DataFrame,
) -> dict[str, Any]:
    pop_total = _measure_value(pop, MEASURE_POPULATION, code)
    age_values = [_measure_value(pop, measure, code) for measure in MEASURES_AGE_65_PLUS]
    age_65_plus = (
        sum(age_values) if all(pd.notna(val) for val in age_values) else float("nan")
    )

    share_65 = float("nan")
    if pd.notna(pop_total) and pop_total > 0 and pd.notna(age_65_plus):
        share_65 = age_65_plus / pop_total

    median_income = _measure_value(inc, MEASURE_MEDIAN_INCOME, code)
    adults = _measure_value(edu, MEASURE_AGE_18_PLUS, code)
    higher = _measure_value(edu, MEASURE_HIGHER_EDUCATION, code)
    share_higher = float("nan")
    if pd.notna(adults) and adults > 0 and pd.notna(higher):
        share_higher = higher / adults

    base: dict[str, Any] = {
        "postal_code": str(code).zfill(5) if code != NATIONAL_POSTAL_CODE else code,
        "data_year": int(data_year),
        "population": pop_total if pd.notna(pop_total) else float("nan"),
        "median_income_eur": median_income,
        "share_age_65_plus": share_65,
        "share_higher_education": share_higher,
        "households_total": _measure_value(households, MEASURE_HOUSEHOLDS_TOTAL, code),
        "average_household_size": _measure_value(
            households, MEASURE_AVG_HOUSEHOLD_SIZE, code
        ),
        "average_floor_area_per_person": _measure_value(
            households, MEASURE_AVG_FLOOR_AREA_PER_PERSON, code
        ),
        "households_owner_occupied": _measure_value(
            households, MEASURE_HOUSEHOLDS_OWNER, code
        ),
        "households_rented": _measure_value(households, MEASURE_HOUSEHOLDS_RENTED, code),
        "median_household_income_eur": _measure_value(
            household_income, MEASURE_MEDIAN_HOUSEHOLD_INCOME, code
        ),
        "dwellings": _measure_value(dwellings, MEASURE_DWELLINGS, code),
        "average_floor_area_per_dwelling": _measure_value(
            dwellings, MEASURE_AVG_FLOOR_AREA_DWELLING, code
        ),
        "dwellings_blocks_of_flats": _measure_value(
            dwellings, MEASURE_DWELLINGS_BLOCKS, code
        ),
        "dwellings_small_houses": _measure_value(
            dwellings, MEASURE_DWELLINGS_SMALL_HOUSES, code
        ),
        "activity_inhabitants": _measure_value(
            activity, MEASURE_ACTIVITY_INHABITANTS, code
        ),
        "employed": _measure_value(activity, MEASURE_EMPLOYED, code),
        "unemployed": _measure_value(activity, MEASURE_UNEMPLOYED, code),
        "students": _measure_value(activity, MEASURE_STUDENTS, code),
        "pensioners": _measure_value(activity, MEASURE_PENSIONERS, code),
    }
    base.update(_derived_measures(base))
    return base


def _assemble_demographics(
    population: pd.DataFrame,
    income: pd.DataFrame,
    education: pd.DataFrame,
    households: pd.DataFrame,
    household_income: pd.DataFrame,
    dwellings: pd.DataFrame,
    activity: pd.DataFrame,
    *,
    data_year: str,
) -> pd.DataFrame:
    pop = _pivot_measures(population)
    inc = _pivot_measures(income)
    edu = _pivot_measures(education)
    hh = _pivot_measures(households)
    hh_inc = _pivot_measures(household_income)
    dwell = _pivot_measures(dwellings)
    act = _pivot_measures(activity)

    codes = sorted(
        set(pop.index)
        | set(inc.index)
        | set(edu.index)
        | set(hh.index)
        | set(hh_inc.index)
        | set(dwell.index)
        | set(act.index)
    )
    codes = [c for c in codes if c != NATIONAL_POSTAL_CODE]

    rows = [
        _row_for_postal_code(
            code,
            data_year=data_year,
            pop=pop,
            inc=inc,
            edu=edu,
            households=hh,
            household_income=hh_inc,
            dwellings=dwell,
            activity=act,
        )
        for code in codes
    ]

    frame = pd.DataFrame(rows)
    return _ensure_demographics_columns(frame)


def _assemble_national_demographics(
    population: pd.DataFrame,
    income: pd.DataFrame,
    education: pd.DataFrame,
    households: pd.DataFrame,
    household_income: pd.DataFrame,
    dwellings: pd.DataFrame,
    activity: pd.DataFrame,
    *,
    data_year: str,
) -> pd.Series:
    pop = _pivot_measures(population)
    inc = _pivot_measures(income)
    edu = _pivot_measures(education)
    hh = _pivot_measures(households)
    hh_inc = _pivot_measures(household_income)
    dwell = _pivot_measures(dwellings)
    act = _pivot_measures(activity)
    row = _row_for_postal_code(
        NATIONAL_POSTAL_CODE,
        data_year=data_year,
        pop=pop,
        inc=inc,
        edu=edu,
        households=hh,
        household_income=hh_inc,
        dwellings=dwell,
        activity=act,
    )
    row.pop("postal_code", None)
    return pd.Series(row)[list(NATIONAL_DEMOGRAPHICS_COLUMNS)]


def _fetch_all_tables(
    data_year: str,
    *,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]],
    get_metadata: Callable[[str], dict[str, Any]],
    pause_seconds: float,
    postal_codes: Sequence[str] | None = None,
    include_sss: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    common = {
        "post_json": post_json,
        "get_metadata": get_metadata,
        "pause_seconds": pause_seconds,
        "postal_codes": postal_codes,
        "include_sss": include_sss,
    }
    pop_long = _fetch_table_measures(
        TABLE_POPULATION,
        data_year,
        [MEASURE_POPULATION, *MEASURES_AGE_65_PLUS],
        **common,
    )
    income_long = _fetch_table_measures(
        TABLE_INCOME, data_year, [MEASURE_MEDIAN_INCOME], **common
    )
    edu_long = _fetch_table_measures(
        TABLE_EDUCATION,
        data_year,
        [MEASURE_AGE_18_PLUS, MEASURE_HIGHER_EDUCATION],
        **common,
    )
    households_long = _fetch_table_measures(
        TABLE_HOUSEHOLDS,
        data_year,
        [
            MEASURE_HOUSEHOLDS_TOTAL,
            MEASURE_AVG_HOUSEHOLD_SIZE,
            MEASURE_AVG_FLOOR_AREA_PER_PERSON,
            MEASURE_HOUSEHOLDS_OWNER,
            MEASURE_HOUSEHOLDS_RENTED,
        ],
        **common,
    )
    household_income_long = _fetch_table_measures(
        TABLE_HOUSEHOLD_INCOME,
        data_year,
        [MEASURE_MEDIAN_HOUSEHOLD_INCOME],
        **common,
    )
    dwellings_long = _fetch_table_measures(
        TABLE_DWELLINGS,
        data_year,
        [
            MEASURE_DWELLINGS,
            MEASURE_AVG_FLOOR_AREA_DWELLING,
            MEASURE_DWELLINGS_BLOCKS,
            MEASURE_DWELLINGS_SMALL_HOUSES,
        ],
        **common,
    )
    activity_long = _fetch_table_measures(
        TABLE_ACTIVITY,
        data_year,
        [
            MEASURE_ACTIVITY_INHABITANTS,
            MEASURE_EMPLOYED,
            MEASURE_UNEMPLOYED,
            MEASURE_STUDENTS,
            MEASURE_PENSIONERS,
        ],
        **common,
    )
    return (
        pop_long,
        income_long,
        edu_long,
        households_long,
        household_income_long,
        dwellings_long,
        activity_long,
    )


def fetch_demographics(
    *,
    data_year: str = DEFAULT_DATA_YEAR,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    get_metadata: Callable[[str], dict[str, Any]] | None = None,
    pause_seconds: float = 0.25,
) -> DemographicsBundle:
    """Download Paavo demographics for one reference year."""
    post = post_json or _default_post
    get_meta = get_metadata or _default_get_metadata

    try:
        tables = _fetch_all_tables(
            data_year,
            post_json=post,
            get_metadata=get_meta,
            pause_seconds=pause_seconds,
            include_sss=False,
        )
        national_tables = _fetch_all_tables(
            data_year,
            post_json=post,
            get_metadata=get_meta,
            pause_seconds=pause_seconds,
            postal_codes=[NATIONAL_POSTAL_CODE],
            include_sss=True,
        )
    except requests.RequestException as exc:
        raise PxWebError(f"Could not reach Statistics Finland PxWeb API: {exc}") from exc

    areas = _assemble_demographics(*tables, data_year=data_year)
    national = _assemble_national_demographics(*national_tables, data_year=data_year)
    return DemographicsBundle(areas=areas, national=national)


def join_demographics_to_postal_codes(
    demographics: pd.DataFrame,
    postal_codes: Sequence[str],
) -> DemographicsJoinReport:
    """Left-join demographics onto postal codes; report codes only in one source."""
    demo = _ensure_demographics_columns(demographics.copy())
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


def _ensure_demographics_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in DEMOGRAPHICS_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan") if col != "data_year" else pd.NA
    if "postal_code" in out.columns:
        out["postal_code"] = out["postal_code"].astype(str).str.zfill(5)
    if "data_year" in out.columns:
        out["data_year"] = out["data_year"].astype("Int64")
    return out[list(DEMOGRAPHICS_COLUMNS)]


def _read_demographics_snapshot(path: Path | None = None) -> pd.DataFrame:
    path = path or data_paths.DEMOGRAPHICS_SNAPSHOT_FILE
    frame = pd.read_csv(path, compression="gzip")
    return _ensure_demographics_columns(frame)


def _read_national_demographics_snapshot(path: Path | None = None) -> pd.Series:
    path = path or data_paths.DEMOGRAPHICS_NATIONAL_SNAPSHOT_FILE
    with gzip_open_text(path) as handle:
        payload = json.load(handle)
    series = pd.Series(payload)
    for col in NATIONAL_DEMOGRAPHICS_COLUMNS:
        if col not in series.index:
            series[col] = float("nan") if col != "data_year" else pd.NA
    return series[list(NATIONAL_DEMOGRAPHICS_COLUMNS)]


def gzip_open_text(path: Path):
    import gzip

    return gzip.open(path, "rt", encoding="utf-8")


def _snapshot_demographics_available() -> bool:
    return data_paths.DEMOGRAPHICS_SNAPSHOT_FILE.is_file()


def _snapshot_national_demographics_available() -> bool:
    return data_paths.DEMOGRAPHICS_NATIONAL_SNAPSHOT_FILE.is_file()


def _load_demographics_fixture() -> pd.DataFrame:
    path = data_paths.DEMOGRAPHICS_FIXTURE_FILE
    frame = pd.read_csv(path)
    return _ensure_demographics_columns(frame)


def _load_national_demographics_fixture() -> pd.Series:
    path = data_paths.DEMOGRAPHICS_NATIONAL_FIXTURE_FILE
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    series = pd.Series(payload)
    return series[list(NATIONAL_DEMOGRAPHICS_COLUMNS)]


def attach_demographics_to_summaries(
    summaries: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Add median income (and other demo columns) aligned to summary index."""
    out = summaries.copy()
    demo = _ensure_demographics_columns(demographics)
    demo["postal_code"] = demo["postal_code"].astype(str).str.zfill(5)
    demo_index = demo.set_index("postal_code")
    index_codes = out.index.astype(str).str.zfill(5)
    attach_cols = [
        col
        for col in DEMOGRAPHICS_COLUMNS
        if col != "postal_code" and col in demo_index.columns
    ]
    for col in attach_cols:
        out[col] = demo_index.reindex(index_codes)[col].to_numpy()
    return out


def load_demographics_bundle(*, refresh: bool = False) -> DemographicsBundle:
    """Load area and national demographics from snapshot, cache, or PxWeb."""
    if data_paths.use_fixtures():
        return DemographicsBundle(
            areas=_load_demographics_fixture(),
            national=_load_national_demographics_fixture(),
        )

    if refresh:
        bundle = fetch_demographics()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {"areas": bundle.areas, NATIONAL_CACHE_KEY: bundle.national.to_dict()},
            CACHE_FILE,
        )
        return bundle

    areas: pd.DataFrame | None = None
    national: pd.Series | None = None

    if _snapshot_demographics_available():
        areas = _read_demographics_snapshot()
    if _snapshot_national_demographics_available():
        national = _read_national_demographics_snapshot()

    if areas is not None:
        if national is None:
            national = pd.Series({col: float("nan") for col in NATIONAL_DEMOGRAPHICS_COLUMNS})
            national["data_year"] = areas["data_year"].dropna().iloc[0] if not areas.empty else pd.NA
        return DemographicsBundle(areas=areas, national=national)

    if CACHE_FILE.exists():
        cached = pd.read_pickle(CACHE_FILE)
        if isinstance(cached, dict) and "areas" in cached:
            areas = _ensure_demographics_columns(cached["areas"])
            national = pd.Series(cached.get(NATIONAL_CACHE_KEY, {}))
        else:
            areas = _ensure_demographics_columns(cached)
        national_path = CACHE_DIR / "demographics_national.json"
        if national is None or national.empty:
            if national_path.is_file():
                with national_path.open(encoding="utf-8") as handle:
                    national = pd.Series(json.load(handle))
        if national is not None and len(national):
            national = national[list(NATIONAL_DEMOGRAPHICS_COLUMNS)]
            return DemographicsBundle(areas=areas, national=national)
        if areas is not None:
            return DemographicsBundle(areas=areas, national=pd.Series(dtype=float))

    bundle = fetch_demographics()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {"areas": bundle.areas, NATIONAL_CACHE_KEY: bundle.national.to_dict()},
        CACHE_FILE,
    )
    return bundle


def load_demographics(*, refresh: bool = False) -> pd.DataFrame:
    """Load demographics from snapshot, cache, or PxWeb."""
    return load_demographics_bundle(refresh=refresh).areas


def load_national_demographics(*, refresh: bool = False) -> pd.Series:
    """National (SSS) Paavo comparison row."""
    return load_demographics_bundle(refresh=refresh).national


def write_national_demographics_snapshot(national: pd.Series, path: Path) -> int:
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = national.to_dict()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path.stat().st_size
