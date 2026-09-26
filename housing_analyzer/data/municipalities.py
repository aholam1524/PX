"""Yearly municipality housing prices and merged municipality boundaries."""

from __future__ import annotations

import gzip
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from housing_analyzer.data import paths as data_paths
from housing_analyzer.data.boundaries import DEFAULT_SIMPLIFY_TOLERANCE, _normalize_postal_code
from housing_analyzer.data.cpi import cpi_by_quarter
from housing_analyzer.data.prices import (
    PxWebError,
    _category_codes,
    _category_labels,
    _is_missing,
    _to_float,
    _to_int,
    read_max_cells,
)

logger = logging.getLogger(__name__)

API_URL = (
    "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/ashi/13mx.px"
)

VAR_TIME = "timeperiod_y"
VAR_MUNICIPALITY = "kunta_1_20150101"
VAR_BUILDING = "talotyyppi_5_20111209"
VAR_CONTENTS = "contentscode"

MEASURE_PRICE = "keskihinta_aritm_nw"
MEASURE_TRANSACTIONS = "lkm_julk20"
MEASURE_TRANSACTIONS_LEGACY = "lkm_julk19"
MEASURES = (MEASURE_PRICE, MEASURE_TRANSACTIONS, MEASURE_TRANSACTIONS_LEGACY)

_PRE_2020_YEAR = 2020

_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])")

CACHE_DIR = data_paths.RAW_DIR
CACHE_FILE = CACHE_DIR / "municipality_prices.pkl"


@dataclass(frozen=True)
class MunicipalityJoinReport:
    """Municipality codes present in only one of prices or boundaries."""

    only_in_prices: tuple[str, ...]
    only_in_boundaries: tuple[str, ...]


@dataclass(frozen=True)
class MunicipalityYearChoice:
    """Which calendar year was selected for a map quarter."""

    year: int
    source: str  # "quarter_year" or "latest_available"


def _building_type_display(code: str, label: str | None = None) -> str:
    readable = label or code
    return f"{code} — {readable}"


def _municipality_code(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text.zfill(3)


def municipality_code_from_properties(props: Mapping[str, Any]) -> str | None:
    """Normalize municipality number from Paavo postal boundary properties."""
    raw = props.get("kuntanro")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = props.get("kunta")
    return _municipality_code(raw)


def postal_codes_missing_municipality(boundaries: Mapping[str, Any]) -> tuple[str, ...]:
    """Postal codes whose boundary feature lacks a usable municipality number."""
    missing: list[str] = []
    for feature in boundaries.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        postal = _normalize_postal_code(
            props.get("postal_code") or props.get("posti_alue") or ""
        )
        if not postal.strip("0"):
            continue
        if municipality_code_from_properties(props) is None:
            missing.append(postal)
    return tuple(sorted(set(missing)))


def parse_municipality_json_stat2(dataset: dict[str, Any]) -> pd.DataFrame:
    """Parse municipality price json-stat2 into a tidy yearly table."""
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

        year = int(str(keyed[VAR_TIME]).strip())
        municipality_code = _municipality_code(keyed[VAR_MUNICIPALITY])
        if municipality_code is None:
            continue
        municipality_name = labels_by_dim[VAR_MUNICIPALITY].get(
            keyed[VAR_MUNICIPALITY], municipality_code
        )
        building_code = keyed[VAR_BUILDING]
        building_label = labels_by_dim[VAR_BUILDING].get(building_code)
        measure = keyed[VAR_CONTENTS]

        row_key = (year, municipality_code, building_code)
        if not rows or rows[-1].get("_key") != row_key:
            rows.append(
                {
                    "_key": row_key,
                    "municipality_code": municipality_code,
                    "municipality_name": municipality_name.strip(),
                    "year": year,
                    "building_type": _building_type_display(building_code, building_label),
                    "price_per_sqm": float("nan"),
                    "transactions": pd.NA,
                }
            )
        current = rows[-1]
        if measure == MEASURE_PRICE:
            current["price_per_sqm"] = _to_float(raw_value)
        elif measure == MEASURE_TRANSACTIONS:
            current["transactions"] = _to_int(raw_value)
        elif measure == MEASURE_TRANSACTIONS_LEGACY:
            if pd.isna(current["transactions"]):
                current["transactions"] = _to_int(raw_value)

    if not rows:
        return _empty_municipality_prices_frame()

    frame = pd.DataFrame(rows).drop(columns=["_key"])
    frame["municipality_code"] = frame["municipality_code"].astype(str).str.zfill(3)
    frame["year"] = frame["year"].astype(int)
    frame["transactions"] = frame["transactions"].astype("Int64")
    return frame[_column_order()]


def _column_order() -> list[str]:
    return [
        "municipality_code",
        "municipality_name",
        "year",
        "building_type",
        "price_per_sqm",
        "transactions",
    ]


def _empty_municipality_prices_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=_column_order())
    frame["transactions"] = frame["transactions"].astype("Int64")
    return frame


def _build_px_query(
    years: Sequence[str],
    municipalities: Sequence[str],
    building_types: Sequence[str],
) -> dict[str, Any]:
    return {
        "query": [
            {
                "code": VAR_TIME,
                "selection": {"filter": "item", "values": list(years)},
            },
            {
                "code": VAR_MUNICIPALITY,
                "selection": {"filter": "item", "values": list(municipalities)},
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


def _default_get_metadata(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise PxWebError(
            f"PxWeb metadata request failed ({response.status_code}): {response.text[:500]}"
        )
    return response.json()


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


def fetch_municipality_prices(
    *,
    api_url: str = API_URL,
    post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    get_metadata: Callable[[str], dict[str, Any]] | None = None,
    pause_seconds: float = 0.0,
) -> pd.DataFrame:
    """Download the full municipality yearly price table from PxWeb."""
    post = post_json or _default_post
    get_meta = get_metadata or _default_get_metadata

    try:
        metadata = get_meta(api_url)
    except requests.RequestException as exc:
        raise PxWebError(f"Could not reach Statistics Finland PxWeb API: {exc}") from exc

    read_max_cells(metadata)  # validate metadata shape when present
    years = list(_variable_by_code(metadata, VAR_TIME)["values"])
    municipalities = list(_variable_by_code(metadata, VAR_MUNICIPALITY)["values"])
    building_types = list(_variable_by_code(metadata, VAR_BUILDING)["values"])

    payload = _build_px_query(years, municipalities, building_types)
    try:
        dataset = post(api_url, payload)
    except PxWebError:
        raise
    except Exception as exc:
        raise PxWebError(f"PxWeb request failed: {exc}") from exc

    if pause_seconds:
        time.sleep(pause_seconds)

    frame = parse_municipality_json_stat2(dataset)
    return frame.sort_values(
        ["year", "municipality_code", "building_type"], ignore_index=True
    )


def _read_municipality_prices_snapshot(path: Path | None = None) -> pd.DataFrame:
    path = path or data_paths.MUNICIPALITY_PRICES_SNAPSHOT_FILE
    frame = pd.read_csv(path, compression="gzip")
    frame["municipality_code"] = frame["municipality_code"].astype(str).str.zfill(3)
    frame["year"] = frame["year"].astype(int)
    frame["transactions"] = frame["transactions"].astype("Int64")
    return frame[_column_order()]


def _load_municipality_prices_fixture() -> pd.DataFrame:
    with data_paths.MUNICIPALITY_PRICES_FIXTURE_FILE.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    return parse_municipality_json_stat2(dataset)


def load_municipality_prices(*, refresh: bool = False) -> pd.DataFrame:
    """Load municipality prices from snapshot, cache, or PxWeb."""
    if data_paths.use_fixtures():
        return _load_municipality_prices_fixture()

    if refresh:
        frame = fetch_municipality_prices()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(CACHE_FILE)
        return frame

    if data_paths.MUNICIPALITY_PRICES_SNAPSHOT_FILE.is_file():
        return _read_municipality_prices_snapshot()

    if CACHE_FILE.exists():
        frame = pd.read_pickle(CACHE_FILE)
        frame["municipality_code"] = frame["municipality_code"].astype(str).str.zfill(3)
        frame["transactions"] = frame["transactions"].astype("Int64")
        return frame[_column_order()]

    frame = fetch_municipality_prices()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(CACHE_FILE)
    return frame


def apply_municipality_names_to_boundaries(
    collection: Mapping[str, Any],
    prices_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fill ``municipality_name`` on boundary features from the price table."""
    if prices_df.empty:
        return dict(collection)

    names = (
        prices_df.drop_duplicates("municipality_code")
        .set_index("municipality_code")["municipality_name"]
        .astype(str)
    )
    features_out: list[dict[str, Any]] = []
    for feature in collection.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = dict(feature.get("properties") or {})
        code = _municipality_code(props.get("municipality_code"))
        if code is not None:
            props["municipality_code"] = code
            if not str(props.get("municipality_name") or "").strip():
                props["municipality_name"] = names.get(code, "")
        features_out.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": feature.get("geometry"),
            }
        )
    return {"type": "FeatureCollection", "features": features_out}


def build_municipality_boundaries(
    boundaries: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_SIMPLIFY_TOLERANCE,
) -> dict[str, Any]:
    """Dissolve postal-code areas into municipality polygons."""
    missing = postal_codes_missing_municipality(boundaries)
    if missing:
        logger.info(
            "Postal codes missing municipality number (%d): %s",
            len(missing),
            ", ".join(missing[:20]) + ("…" if len(missing) > 20 else ""),
        )

    groups: dict[str, list[Any]] = {}
    for feature in boundaries.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        code = municipality_code_from_properties(props)
        if code is None:
            continue
        geom = feature.get("geometry")
        if not geom:
            continue
        try:
            groups.setdefault(code, []).append(shape(geom))
        except (TypeError, ValueError):
            continue

    features_out: list[dict[str, Any]] = []
    for code in sorted(groups):
        geoms = groups[code]
        if not geoms:
            continue
        merged = unary_union(geoms)
        if merged.is_empty:
            continue
        simplified = merged.simplify(tolerance, preserve_topology=True)
        features_out.append(
            {
                "type": "Feature",
                "properties": {
                    "municipality_code": code,
                    "municipality_name": "",
                },
                "geometry": mapping(simplified),
            }
        )

    return {"type": "FeatureCollection", "features": features_out}


def _read_municipality_boundaries_snapshot(path: Path | None = None) -> dict[str, Any]:
    path = path or data_paths.MUNICIPALITY_BOUNDARIES_SNAPSHOT_FILE
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _load_municipality_boundaries_fixture() -> dict[str, Any]:
    with data_paths.MUNICIPALITY_BOUNDARIES_FIXTURE_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_municipality_boundaries(*, refresh: bool = False) -> dict[str, Any]:
    """Load merged municipality boundaries from snapshot or build from postal areas."""
    if data_paths.use_fixtures():
        return _load_municipality_boundaries_fixture()

    if refresh:
        from housing_analyzer.data.boundaries import load_boundaries

        postal = load_boundaries(refresh=True)
        built = build_municipality_boundaries(postal)
        prices = load_municipality_prices(refresh=False)
        return apply_municipality_names_to_boundaries(built, prices)

    if data_paths.MUNICIPALITY_BOUNDARIES_SNAPSHOT_FILE.is_file():
        return _read_municipality_boundaries_snapshot()

    from housing_analyzer.data.boundaries import load_boundaries

    postal = load_boundaries(refresh=False)
    built = build_municipality_boundaries(postal)
    prices = load_municipality_prices(refresh=False)
    return apply_municipality_names_to_boundaries(built, prices)


def _year_slice(
    mun_df: pd.DataFrame,
    year: int,
    building_type: str | None,
) -> pd.DataFrame:
    frame = mun_df.loc[mun_df["year"] == year].copy()
    if building_type is not None:
        bt = str(building_type)
        frame = frame.loc[
            frame["building_type"].eq(bt)
            | frame["building_type"].str.startswith(f"{bt} —")
            | frame["building_type"].str.startswith(f"{bt} -")
        ]
    if frame.empty:
        return frame
    frame["municipality_code"] = frame["municipality_code"].astype(str).str.zfill(3)
    frame = frame.sort_values(["municipality_code", "building_type"], kind="stable")
    return frame.drop_duplicates(subset=["municipality_code"], keep="first")


def municipality_year_for_quarter(
    quarter: str,
    available_years: Sequence[int],
) -> MunicipalityYearChoice:
    """Pick the price year for a map quarter."""
    match = _QUARTER_RE.match(quarter.strip())
    if not match:
        raise ValueError(f"Invalid quarter label: {quarter!r}")
    quarter_year = int(match.group(1))
    years = sorted({int(y) for y in available_years})
    if quarter_year in years:
        return MunicipalityYearChoice(year=quarter_year, source="quarter_year")
    if not years:
        raise ValueError("available_years is empty")
    fallback = years[-1]
    return MunicipalityYearChoice(year=fallback, source="latest_available")


def _pct_change(current: float, prior: float) -> float:
    if pd.isna(current) or pd.isna(prior) or prior == 0.0:
        return float("nan")
    return (float(current) / float(prior) - 1.0) * 100.0


def _reliability_labels(
    mun_df: pd.DataFrame,
    current: pd.DataFrame,
    year: int,
    building_type: str | None,
    *,
    min_transactions: int = 10,
    window_years: int = 4,
) -> pd.Series:
    """Reliability label for every municipality in ``current``, computed in one pass.

    Same rules as the old per-municipality ``_reliability_label``, but the
    window years are sliced once each (not once per municipality), since
    ``_year_slice`` re-filters/sorts/dedupes the full multi-year frame.
    """
    if year < _PRE_2020_YEAR:
        return pd.Series("unknown", index=current.index, dtype=object)

    codes = current["municipality_code"]
    totals = pd.Series(0.0, index=codes.to_numpy())
    for offset in range(window_years):
        chunk = _year_slice(mun_df, year - offset, building_type)
        if chunk.empty:
            continue
        tx = chunk.set_index("municipality_code")["transactions"].astype(float)
        totals = totals.add(tx.reindex(totals.index).fillna(0.0), fill_value=0.0)

    row_totals = totals.reindex(codes.to_numpy()).to_numpy()
    labels = np.where(
        current["price_per_sqm"].isna().to_numpy(),
        "none",
        np.where(row_totals >= min_transactions, "ok", "low"),
    )
    return pd.Series(labels, index=current.index, dtype=object)


def municipality_metrics(
    mun_df: pd.DataFrame,
    year: int,
    building_type: str | None = None,
    *,
    min_transactions: int = 10,
    window_years: int = 4,
) -> pd.DataFrame:
    """Per-municipality price, nominal changes, transactions and reliability."""
    current = _year_slice(mun_df, year, building_type)
    if current.empty:
        return pd.DataFrame(
            columns=[
                "municipality_code",
                "municipality_name",
                "price_per_sqm",
                "pct_change_1y",
                "pct_change_5y",
                "transactions",
                "reliability",
            ]
        )

    prior_1y = _year_slice(mun_df, year - 1, building_type).set_index("municipality_code")[
        "price_per_sqm"
    ]
    prior_5y = _year_slice(mun_df, year - 5, building_type).set_index("municipality_code")[
        "price_per_sqm"
    ]

    reliability = _reliability_labels(
        mun_df,
        current,
        year,
        building_type,
        min_transactions=min_transactions,
        window_years=window_years,
    )

    rows: list[dict[str, Any]] = []
    for idx, row in current.iterrows():
        code = row["municipality_code"]
        price = float(row["price_per_sqm"])
        rows.append(
            {
                "municipality_code": code,
                "municipality_name": row["municipality_name"],
                "price_per_sqm": price,
                "pct_change_1y": _pct_change(price, prior_1y.get(code, float("nan"))),
                "pct_change_5y": _pct_change(price, prior_5y.get(code, float("nan"))),
                "transactions": row["transactions"],
                "reliability": reliability.loc[idx],
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["transactions"] = frame["transactions"].astype("Int64")
    return frame.sort_values("municipality_code", ignore_index=True)


def cpi_by_year(cpi_df: pd.DataFrame) -> pd.Series:
    """Calendar-year CPI as the mean of available quarterly CPI in that year."""
    quarterly = cpi_by_quarter(cpi_df)
    if quarterly.empty:
        return pd.Series(dtype=float)

    buckets: dict[int, list[float]] = {}
    for quarter, value in quarterly.items():
        if pd.isna(value):
            continue
        match = _QUARTER_RE.match(str(quarter))
        if not match:
            continue
        year = int(match.group(1))
        buckets.setdefault(year, []).append(float(value))

    rows = {year: sum(values) / len(values) for year, values in buckets.items() if values}
    series = pd.Series(rows, dtype=float)
    series.index.name = "year"
    series.name = "cpi"
    return series.sort_index()


def municipality_real_changes(
    mun_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    year: int,
    building_type: str | None = None,
    *,
    base_year: int | None = None,
) -> pd.DataFrame:
    """Add real (CPI-deflated) prices and 1y/5y real percentage changes."""
    metrics = municipality_metrics(mun_df, year, building_type)
    if metrics.empty:
        return metrics.assign(
            real_price_per_sqm=float("nan"),
            pct_change_1y_real=float("nan"),
            pct_change_5y_real=float("nan"),
        )

    yearly_cpi = cpi_by_year(cpi_df)
    if yearly_cpi.empty:
        out = metrics.copy()
        out["real_price_per_sqm"] = float("nan")
        out["pct_change_1y_real"] = float("nan")
        out["pct_change_5y_real"] = float("nan")
        return out

    base = base_year if base_year is not None else year
    if base not in yearly_cpi.index or pd.isna(yearly_cpi.loc[base]) or yearly_cpi.loc[base] == 0:
        out = metrics.copy()
        out["real_price_per_sqm"] = float("nan")
        out["pct_change_1y_real"] = float("nan")
        out["pct_change_5y_real"] = float("nan")
        return out

    base_cpi = float(yearly_cpi.loc[base])
    year_cpi = yearly_cpi.reindex([year]).iloc[0]
    factor = base_cpi / float(year_cpi) if pd.notna(year_cpi) and year_cpi != 0 else float("nan")

    real_now = metrics["price_per_sqm"].astype(float) * factor
    real_now.loc[metrics["price_per_sqm"].isna() | pd.isna(factor)] = float("nan")

    def _real_price_at(target_year: int) -> pd.Series:
        slice_y = _year_slice(mun_df, target_year, building_type).set_index("municipality_code")
        nominal = slice_y["price_per_sqm"].astype(float)
        cpi_val = yearly_cpi.get(target_year, float("nan"))
        if pd.isna(cpi_val) or cpi_val == 0:
            return pd.Series(float("nan"), index=nominal.index)
        f = base_cpi / float(cpi_val)
        return nominal * f

    real_prior_1y = _real_price_at(year - 1)
    real_prior_5y = _real_price_at(year - 5)

    out = metrics.copy()
    out["real_price_per_sqm"] = real_now.to_numpy()
    out["pct_change_1y_real"] = [
        _pct_change(r_now, real_prior_1y.get(code, float("nan")))
        for code, r_now in zip(out["municipality_code"], out["real_price_per_sqm"], strict=False)
    ]
    out["pct_change_5y_real"] = [
        _pct_change(r_now, real_prior_5y.get(code, float("nan")))
        for code, r_now in zip(out["municipality_code"], out["real_price_per_sqm"], strict=False)
    ]
    return out


def municipality_codes_from_boundaries(
    municipality_boundaries: Mapping[str, Any],
) -> tuple[str, ...]:
    """Municipality numbers present in a municipality boundaries FeatureCollection."""
    codes: set[str] = set()
    for feature in municipality_boundaries.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        code = _municipality_code(props.get("municipality_code"))
        if code is not None:
            codes.add(code)
    return tuple(sorted(codes))


def municipality_join_report(
    mun_df: pd.DataFrame,
    municipality_boundaries: Mapping[str, Any],
) -> MunicipalityJoinReport:
    """List municipality codes that appear in only prices or only boundaries."""
    price_codes = set()
    if not mun_df.empty:
        price_codes = set(mun_df["municipality_code"].astype(str).str.zfill(3))

    boundary_codes = set(municipality_codes_from_boundaries(municipality_boundaries))

    only_in_prices = tuple(sorted(price_codes - boundary_codes))
    only_in_boundaries = tuple(sorted(boundary_codes - price_codes))

    if only_in_prices or only_in_boundaries:
        logger.info(
            "Municipality codes only in prices (%d): %s",
            len(only_in_prices),
            ", ".join(only_in_prices[:20]) + ("…" if len(only_in_prices) > 20 else ""),
        )
        logger.info(
            "Municipality codes only in boundaries (%d): %s",
            len(only_in_boundaries),
            ", ".join(only_in_boundaries[:20])
            + ("…" if len(only_in_boundaries) > 20 else ""),
        )

    return MunicipalityJoinReport(
        only_in_prices=only_in_prices,
        only_in_boundaries=only_in_boundaries,
    )
