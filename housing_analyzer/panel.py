"""Area detail panel helpers: trends, comparisons, flags, and charts (no Streamlit)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from housing_analyzer.analysis.market_activity import (
    market_activity,
    market_activity_reliability_note,
)
from housing_analyzer.analysis.metrics import (
    area_prices_at,
    quarter_index,
    regional_average,
    to_real,
)
from housing_analyzer.data.cpi import cpi_by_quarter, load_cpi
from housing_analyzer.map import (
    BUILDING_TYPE_CHOICES,
    latest_quarter_with_data,
    resolve_building_type_label,
    trailing_sales_sum,
)

DEFAULT_SELECTED_POSTAL_CODE = "00100"

MUNICIPALITY_CODE_PROPERTY = "kunta"
NATIONAL_GROUP = "__national__"
SALES_VOLUME_FROM_QUARTER = "2020Q1"
_UNUSUAL_MIN_QUARTERS = 12
_UNUSUAL_N_STD = 3.0

_MUNICIPALITY_SUFFIX = re.compile(r"\(([^)]+)\)\s*$")


@dataclass(frozen=True)
class TrendChartData:
    """Quarterly price series for an area and optional comparisons."""

    quarters: tuple[str, ...]
    area_prices: tuple[float | None, ...]
    municipality_prices: tuple[float | None, ...] | None
    national_prices: tuple[float | None, ...]
    municipality_label: str | None
    municipality_available: bool
    indexed: bool
    use_real: bool


def boundary_properties(
    boundaries: Mapping[str, Any], postal_code: str
) -> dict[str, Any]:
    """Properties dict for one postal code from boundary GeoJSON, if present."""
    code = str(postal_code).zfill(5)
    for feature in boundaries.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        feat_code = str(props.get("postal_code") or props.get("posti_alue", "")).zfill(
            5
        )
        if feat_code == code:
            return dict(props)
    return {}


def municipality_code_from_boundaries(
    boundaries: Mapping[str, Any], postal_code: str
) -> str | None:
    props = boundary_properties(boundaries, postal_code)
    raw = props.get(MUNICIPALITY_CODE_PROPERTY)
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    return text or None


def municipality_name_from_prices(
    prices_df: pd.DataFrame, postal_code: str
) -> str | None:
    """Parse municipality from price area labels like ``… (Helsinki)``."""
    code = str(postal_code).zfill(5)
    names = (
        prices_df.loc[prices_df["postal_code"].astype(str).str.zfill(5) == code][
            "area_name"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    for name in names:
        match = _MUNICIPALITY_SUFFIX.search(name.strip())
        if match:
            return match.group(1).strip()
    return None


def area_display_name(prices_df: pd.DataFrame, postal_code: str) -> str:
    code = str(postal_code).zfill(5)
    rows = prices_df.loc[
        prices_df["postal_code"].astype(str).str.zfill(5) == code, "area_name"
    ]
    if rows.empty:
        return ""
    name = str(rows.iloc[0])
    return _MUNICIPALITY_SUFFIX.sub("", name).strip()


def postal_code_name_lookup(prices_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Precompute postal_code -> (area_display_name, municipality) once.

    Equivalent to calling ``area_display_name`` and
    ``municipality_name_from_prices`` per postal code, but scans
    ``prices_df`` a single time instead of once per code.
    """
    codes = prices_df["postal_code"].astype(str).str.zfill(5)
    lookup: dict[str, tuple[str, str]] = {}
    for code, area_names in prices_df["area_name"].groupby(codes):
        names = area_names.dropna().astype(str)
        if names.empty:
            lookup[code] = ("", "")
            continue
        display_name = _MUNICIPALITY_SUFFIX.sub("", str(names.iloc[0])).strip()
        municipality = ""
        for name in names.unique():
            match = _MUNICIPALITY_SUFFIX.search(name.strip())
            if match:
                municipality = match.group(1).strip()
                break
        lookup[code] = (display_name, municipality)
    return lookup


def _header_text_part(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return text


def format_area_header(
    postal_code: Any,
    area_name: Any,
    municipality: Any,
) -> str:
    """Single markdown line: bold postal code, area name, municipality after a comma."""
    code_raw = _header_text_part(postal_code)
    code = code_raw.zfill(5) if code_raw else ""
    name = _header_text_part(area_name)
    muni = _header_text_part(municipality)

    show_muni = bool(muni)
    if show_muni and name:
        if name.casefold() == muni.casefold():
            show_muni = False
        elif name.casefold().endswith(muni.casefold()):
            show_muni = False

    line = ""
    if code:
        line = f"**{code}**"
    if name:
        line = f"{line} {name}".strip() if line else name
    if show_muni:
        line = f"{line}, {muni}" if line else muni
    return line


def postal_to_municipality_group(boundaries: Mapping[str, Any]) -> dict[str, str]:
    """Map each boundary postal code to its municipality code (``kunta``)."""
    mapping: dict[str, str] = {}
    for feature in boundaries.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        code = str(props.get("postal_code") or props.get("posti_alue", "")).zfill(5)
        muni = props.get(MUNICIPALITY_CODE_PROPERTY)
        if code and muni is not None and str(muni).strip():
            mapping[code] = str(muni).strip()
    return mapping


def national_group_mapping(postal_codes: Any) -> dict[str, str]:
    return {str(code).zfill(5): NATIONAL_GROUP for code in postal_codes}


def _sorted_quarters(quarters: Any) -> list[str]:
    return sorted(set(quarters), key=quarter_index)


def _series_to_tuple(series: pd.Series, quarters: list[str]) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for q in quarters:
        value = series.get(q, float("nan"))
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(
            value
        ):
            out.append(None)
        else:
            out.append(float(value))
    return tuple(out)


def quarterly_area_prices(
    df: pd.DataFrame,
    postal_code: str,
    building_type: str | None = None,
    *,
    price_column: str = "price_per_sqm",
) -> pd.Series:
    """Price per m² by quarter for one postal code (NaN where missing)."""
    code = str(postal_code).zfill(5)
    quarters = _sorted_quarters(df["quarter"].unique())
    values: dict[str, float] = {}
    for quarter in quarters:
        prices = area_prices_at(df, quarter, building_type, price_column=price_column)
        if code in prices.index:
            values[quarter] = float(prices.loc[code, price_column])
        else:
            values[quarter] = float("nan")
    return pd.Series(values)


def quarterly_group_average(
    df: pd.DataFrame,
    group: Mapping[str, str],
    group_name: str,
    building_type: str | None = None,
    *,
    price_column: str = "price_per_sqm",
) -> pd.Series:
    """One group's average price per m² for every quarter in ``df``."""
    quarters = _sorted_quarters(df["quarter"].unique())
    values: dict[str, float] = {}
    for quarter in quarters:
        avg = regional_average(
            df, group, quarter, building_type=building_type, price_column=price_column
        )
        if avg.empty:
            values[quarter] = float("nan")
            continue
        match = avg.loc[avg["group"] == group_name, price_column]
        values[quarter] = float(match.iloc[0]) if len(match) else float("nan")
    return pd.Series(values)


def index_series_to_100(series: pd.Series) -> pd.Series:
    """Rebase the first non-null observation to 100; missing stays missing."""
    out = series.astype(float).copy()
    valid = out.dropna()
    if valid.empty:
        return out
    base = float(valid.iloc[0])
    if base == 0.0 or np.isnan(base):
        return pd.Series(float("nan"), index=out.index)
    return (out / base) * 100.0


def quarter_on_quarter_changes(prices: pd.Series) -> pd.Series:
    """Quarter-on-quarter % change; first quarter and gaps are NaN."""
    ordered = _sorted_quarters(prices.index)
    changes: dict[str, float] = {}
    prev_q: str | None = None
    prev_price: float | None = None
    for quarter in ordered:
        price = prices.get(quarter, float("nan"))
        if (
            prev_q is not None
            and prev_price is not None
            and pd.notna(prev_price)
            and prev_price != 0
            and pd.notna(price)
        ):
            changes[quarter] = (float(price) / float(prev_price) - 1.0) * 100.0
        else:
            changes[quarter] = float("nan")
        if pd.notna(price):
            prev_q = quarter
            prev_price = float(price)
    return pd.Series(changes)


def flag_unusual_quarter_changes(
    prices: pd.Series,
    *,
    min_quarters: int = _UNUSUAL_MIN_QUARTERS,
    n_std: float = _UNUSUAL_N_STD,
) -> list[dict[str, Any]]:
    """Quarters whose QoQ change exceeds ``n_std`` times the series' own std."""
    if len(_sorted_quarters(prices.index)) < min_quarters:
        return []

    qoq = quarter_on_quarter_changes(prices).dropna()
    if len(qoq) < 2:
        return []

    values = qoq.to_numpy(dtype=float)
    std = float(np.std(values, ddof=1))
    if std == 0.0 or np.isnan(std):
        return []

    mean = float(np.mean(values))
    threshold = n_std * std
    flagged: list[dict[str, Any]] = []
    for quarter, change in qoq.items():
        if abs(float(change) - mean) > threshold:
            flagged.append(
                {
                    "quarter": str(quarter),
                    "pct_change_qoq": float(change),
                    "mean_qoq": mean,
                    "std_qoq": std,
                    "threshold_qoq": threshold,
                }
            )
    return flagged


def build_trend_chart_data(
    prices_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    postal_code: str,
    building_type: str | None,
    *,
    index_to_100: bool = False,
    use_real: bool = False,
    cpi_df: pd.DataFrame | None = None,
) -> TrendChartData:
    """Area, municipality, and national quarterly price series for the trend chart."""
    code = str(postal_code).zfill(5)
    work_df = prices_df
    price_column = "price_per_sqm"
    if use_real:
        cpi = cpi_df if cpi_df is not None else load_cpi()
        work_df = to_real(prices_df, cpi_by_quarter(cpi))
        price_column = "real_price_per_sqm"

    area = quarterly_area_prices(
        work_df, code, building_type, price_column=price_column
    )
    quarters = _sorted_quarters(area.index)

    muni_map = postal_to_municipality_group(boundaries)
    muni_code = muni_map.get(code)
    municipality_available = muni_code is not None and code in muni_map
    municipality_label = municipality_name_from_prices(prices_df, code)
    if municipality_label is None and muni_code is not None:
        municipality_label = f"Municipality {muni_code}"

    national_map = national_group_mapping(prices_df["postal_code"].unique())
    national = quarterly_group_average(
        work_df,
        national_map,
        NATIONAL_GROUP,
        building_type,
        price_column=price_column,
    )

    municipality: pd.Series | None = None
    if municipality_available and muni_code is not None:
        municipality = quarterly_group_average(
            work_df,
            muni_map,
            muni_code,
            building_type,
            price_column=price_column,
        )

    if index_to_100:
        area = index_series_to_100(area)
        national = index_series_to_100(national)
        if municipality is not None:
            municipality = index_series_to_100(municipality)

    return TrendChartData(
        quarters=tuple(quarters),
        area_prices=_series_to_tuple(area, quarters),
        municipality_prices=(
            _series_to_tuple(municipality, quarters) if municipality is not None else None
        ),
        national_prices=_series_to_tuple(national, quarters),
        municipality_label=municipality_label,
        municipality_available=municipality_available,
        indexed=index_to_100,
        use_real=use_real,
    )


def _plotly_y(values: tuple[float | None, ...]) -> list[float | None]:
    return [None if v is None else float(v) for v in values]


def build_trend_figure(data: TrendChartData, *, area_label: str) -> go.Figure:
    """Plotly line chart with gaps for missing quarters."""
    fig = go.Figure()
    x = list(data.quarters)
    if data.indexed:
        y_unit = "Index (100 = start)"
    elif data.use_real:
        y_unit = "Real EUR/m²"
    else:
        y_unit = "EUR/m²"

    fig.add_trace(
        go.Scatter(
            x=x,
            y=_plotly_y(data.area_prices),
            mode="lines+markers",
            name=area_label,
            connectgaps=False,
        )
    )
    if data.municipality_available and data.municipality_prices is not None:
        label = data.municipality_label or "Municipality average"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=_plotly_y(data.municipality_prices),
                mode="lines",
                name=label,
                connectgaps=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_plotly_y(data.national_prices),
            mode="lines",
            name="National average",
            connectgaps=False,
        )
    )
    fig.update_layout(
        yaxis_title=y_unit,
        xaxis_title="Quarter",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        height=360,
    )
    return fig


def quarterly_transaction_counts(
    df: pd.DataFrame,
    postal_code: str,
    building_type: str | None = None,
    *,
    from_quarter: str = SALES_VOLUME_FROM_QUARTER,
) -> pd.Series:
    """Transaction counts per quarter from ``from_quarter`` onward (NaN if unknown)."""
    code = str(postal_code).zfill(5)
    area = df.loc[df["postal_code"].astype(str).str.zfill(5) == code].copy()
    if building_type is not None:
        area = area.loc[area["building_type"] == building_type]
    if area.empty:
        return pd.Series(dtype=float)

    from_idx = quarter_index(from_quarter)
    area = area.loc[area["quarter"].map(quarter_index) >= from_idx]
    if area.empty:
        return pd.Series(dtype=float)

    grouped = area.groupby("quarter", sort=False)["transactions"].apply(
        lambda col: col.dropna().astype(float).sum(min_count=1)
    )
    quarters = _sorted_quarters(grouped.index)
    return grouped.reindex(quarters)


def build_sales_volume_figure(
    counts: pd.Series,
    *,
    from_quarter: str = SALES_VOLUME_FROM_QUARTER,
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=list(counts.index),
            y=[None if pd.isna(v) else float(v) for v in counts.to_numpy()],
        )
    )
    fig.update_layout(
        title=f"Sales per quarter (from {from_quarter})",
        yaxis_title="Transactions",
        xaxis_title="Quarter",
        height=220,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def area_detail_export_frame(
    df: pd.DataFrame,
    postal_code: str,
    building_type: str | None = None,
) -> pd.DataFrame:
    """Tabular history for CSV download."""
    code = str(postal_code).zfill(5)
    area = df.loc[df["postal_code"].astype(str).str.zfill(5) == code].copy()
    if building_type is not None:
        area = area.loc[area["building_type"] == building_type]
    if area.empty:
        return pd.DataFrame(
            columns=[
                "postal_code",
                "area_name",
                "quarter",
                "building_type",
                "price_per_sqm",
                "transactions",
            ]
        )
    area = area.sort_values(["quarter", "building_type"], kind="stable")
    return area[
        [
            "postal_code",
            "area_name",
            "quarter",
            "building_type",
            "price_per_sqm",
            "transactions",
        ]
    ]


def building_type_panel_options() -> tuple[tuple[str, str], ...]:
    return BUILDING_TYPE_CHOICES


def resolve_panel_building_type(
    prices_df: pd.DataFrame, building_type_code: str | None
) -> str | None:
    if building_type_code is None or building_type_code == "all":
        return None
    return resolve_building_type_label(prices_df, building_type_code)


def trailing_sales_count(
    prices_df: pd.DataFrame,
    postal_code: str,
    quarter: str,
    building_type_code: str | None,
) -> float:
    code = None if building_type_code in (None, "all") else building_type_code
    return trailing_sales_sum(prices_df, postal_code, quarter, code)


MARKET_ACTIVITY_METRIC_HELP = (
    "Sales in the last four quarters per 1,000 inhabitants in the postal area "
    "(Paavo population). A rough activity index, not a turnover rate of the housing stock."
)


def market_activity_for_area(
    prices_df: pd.DataFrame,
    postal_code: str,
    quarter: str,
    building_type_code: str | None,
    demographics_df: pd.DataFrame,
) -> float:
    code = str(postal_code).zfill(5)
    sales_4q = trailing_sales_count(prices_df, code, quarter, building_type_code)
    demo = demographics_df.copy()
    demo["postal_code"] = demo["postal_code"].astype(str).str.zfill(5)
    demo_index = demo.set_index("postal_code")
    population = (
        float(demo_index.loc[code, "population"])
        if code in demo_index.index
        else float("nan")
    )
    return market_activity(sales_4q, population)


def market_activity_panel_caption(sales_4q: float) -> str | None:
    return market_activity_reliability_note(sales_4q)


def default_selected_postal_code(prices_df: pd.DataFrame) -> str | None:
    """Postal code to select on first load: 00100 when present, else busiest in latest quarter."""
    if prices_df.empty:
        return None
    codes = prices_df["postal_code"].astype(str).str.zfill(5)
    if codes.eq(DEFAULT_SELECTED_POSTAL_CODE).any():
        return DEFAULT_SELECTED_POSTAL_CODE
    quarter = latest_quarter_with_data(prices_df)
    if quarter is None:
        return None
    quarter_rows = prices_df.loc[prices_df["quarter"] == quarter]
    if quarter_rows.empty:
        return None
    q_codes = quarter_rows["postal_code"].astype(str).str.zfill(5)
    totals = quarter_rows.groupby(q_codes)["transactions"].sum(min_count=1)
    if totals.empty or totals.isna().all():
        return None
    return str(totals.idxmax())


PAAVO_SUPPRESSED_TOOLTIP = (
    "Statistics Finland suppresses values when counts are too small to publish."
)


@dataclass(frozen=True)
class AreaProfileItem:
    label: str
    area_text: str
    national_text: str
    help: str | None


def _format_profile_number(value: float | None, *, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _format_profile_percent(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "—"
    return f"{int(round(float(value) * 100))}%"


def _national_suffix(national_text: str) -> str:
    if national_text == "—":
        return "(country —)"
    return f"(country {national_text})"


def build_area_profile_items(
    area_row: Mapping[str, Any],
    national_row: Mapping[str, Any],
) -> tuple[AreaProfileItem, ...]:
    """Rows for the detail-panel area profile (Paavo vs national SSS)."""
    tooltip = PAAVO_SUPPRESSED_TOOLTIP

    def item(
        label: str,
        area_key: str,
        *,
        percent: bool = False,
        decimals: int = 1,
    ) -> AreaProfileItem:
        raw_area = area_row.get(area_key)
        raw_nat = national_row.get(area_key)
        formatter = _format_profile_percent if percent else _format_profile_number
        area_text = formatter(raw_area)
        national_text = formatter(raw_nat)
        help_text = tooltip if area_text == "—" else None
        return AreaProfileItem(
            label=label,
            area_text=area_text,
            national_text=_national_suffix(national_text),
            help=help_text,
        )

    return (
        item("Average household size", "average_household_size", decimals=2),
        item("Average floor area per dwelling (m²)", "average_floor_area_per_dwelling"),
        item("Rented households", "rented_share", percent=True),
        item("Unemployment rate", "unemployment_rate", percent=True),
        item("Students (share of inhabitants)", "student_share", percent=True),
        item("Pensioners (share of inhabitants)", "pensioner_share", percent=True),
        item(
            "Dwellings in blocks of flats",
            "share_blocks_of_flats",
            percent=True,
        ),
    )


def area_profile_data_year(
    area_row: Mapping[str, Any], national_row: Mapping[str, Any]
) -> int | None:
    for row in (area_row, national_row):
        year = row.get("data_year")
        if year is not None and not (isinstance(year, float) and np.isnan(year)) and not pd.isna(
            year
        ):
            return int(year)
    return None


def reliability_explanation(
    reliability_label: str | None, sales_4q: float
) -> str:
    if reliability_label == "ok":
        prefix = "OK — enough sales."
    elif reliability_label == "low":
        prefix = "Low — based on few sales."
    elif reliability_label == "none":
        prefix = "No published price."
    elif reliability_label == "unknown":
        prefix = "Unknown (pre-2020)."
    else:
        prefix = "Reliability unknown."
    if sales_4q == sales_4q and not np.isnan(sales_4q):
        n = int(round(sales_4q))
        return f"{prefix} Based on {n} sales in the last 4 quarters."
    return f"{prefix} Based on N sales in the last 4 quarters."
