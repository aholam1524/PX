"""Compare-tab helpers: summary table and multi-area price chart (no Streamlit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from housing_analyzer.analysis.metrics import quarter_index, summarize_areas, to_real
from housing_analyzer.data.cpi import cpi_by_quarter
from housing_analyzer.map import trailing_sales_by_area
from housing_analyzer.panel import index_series_to_100, quarterly_area_prices


def area_catalog(prices_df: pd.DataFrame) -> pd.DataFrame:
    """One row per postal code with display name for search and multiselect."""
    if prices_df.empty:
        return pd.DataFrame(columns=["postal_code", "area_name"])
    grouped = (
        prices_df.assign(
            postal_code=prices_df["postal_code"].astype(str).str.zfill(5)
        )
        .groupby("postal_code", sort=True)["area_name"]
        .first()
    )
    out = grouped.reset_index()
    out["area_name"] = out["area_name"].fillna("").astype(str)
    return out


def search_area_catalog(catalog: pd.DataFrame, query: str) -> list[str]:
    """Postal codes whose code or name contains ``query`` (case-insensitive)."""
    text = query.strip().lower()
    if not text or catalog.empty:
        return []
    codes = catalog["postal_code"].astype(str).str.zfill(5)
    names = catalog["area_name"].astype(str).str.lower()
    mask = codes.str.contains(text, regex=False) | names.str.contains(text, regex=False)
    return sorted(set(codes[mask]))


def format_optional_float(value: Any, *, suffix: str = "", decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "—"
    if suffix == "%":
        return f"{float(value):+.{decimals}f}%"
    if suffix == "eur":
        return f"{float(value):,.0f} EUR/m²"
    if suffix == "int":
        return str(int(round(float(value))))
    return str(value)


def build_comparison_table(
    summaries: pd.DataFrame,
    sales_by_area: pd.Series,
    postal_codes: Sequence[str],
    *,
    include_real: bool = True,
) -> pd.DataFrame:
    """Metric rows and one column per selected postal code (area label as column name)."""
    codes = [_normalize_code(c) for c in postal_codes]
    columns: dict[str, list[str]] = {}
    for code in codes:
        columns[code] = _column_cells(
            summaries, sales_by_area, code, include_real=include_real
        )

    rows = _comparison_row_labels(include_real=include_real)
    return pd.DataFrame(columns, index=rows)


def _normalize_code(postal_code: str) -> str:
    return str(postal_code).zfill(5)


def _comparison_row_labels(*, include_real: bool) -> list[str]:
    rows = [
        "Price per m²",
        "1-year change (nominal)",
        "5-year change (nominal)",
    ]
    if include_real:
        rows.extend(
            [
                "1-year change (real)",
                "5-year change (real)",
            ]
        )
    rows.extend(
        [
            "Sales (last 4 quarters)",
            "Reliability",
            "Rank",
            "Percentile",
        ]
    )
    return rows


def _column_cells(
    summaries: pd.DataFrame,
    sales_by_area: pd.Series,
    code: str,
    *,
    include_real: bool,
) -> list[str]:
    if code not in summaries.index:
        missing_rows = len(_comparison_row_labels(include_real=include_real))
        return ["—"] * missing_rows

    row = summaries.loc[code]
    cells = [
        format_optional_float(row.get("price_per_sqm"), suffix="eur"),
        format_optional_float(row.get("pct_change_1y"), suffix="%"),
        format_optional_float(row.get("pct_change_5y"), suffix="%"),
    ]
    if include_real:
        cells.extend(
            [
                format_optional_float(row.get("pct_change_1y_real"), suffix="%"),
                format_optional_float(row.get("pct_change_5y_real"), suffix="%"),
            ]
        )
    sales = sales_by_area.get(code, float("nan"))
    cells.append(format_optional_float(sales, suffix="int"))
    rel = row.get("reliability")
    cells.append("—" if rel is None or (isinstance(rel, float) and np.isnan(rel)) else str(rel))
    rank = row.get("rank")
    cells.append("—" if pd.isna(rank) else str(int(rank)))
    cells.append(format_optional_float(row.get("percentile"), suffix="int"))
    return cells


@dataclass(frozen=True)
class CompareChartData:
    quarters: tuple[str, ...]
    series: tuple[tuple[str, tuple[float | None, ...]], ...]
    indexed: bool
    use_real: bool


def build_compare_chart_data(
    prices_df: pd.DataFrame,
    postal_codes: Sequence[str],
    building_type: str | None,
    *,
    index_to_100: bool = False,
    use_real: bool = False,
    cpi_quarterly: pd.Series | None = None,
) -> CompareChartData:
    work_df = prices_df
    price_column = "price_per_sqm"
    if use_real and cpi_quarterly is not None:
        work_df = to_real(prices_df, cpi_quarterly)
        price_column = "real_price_per_sqm"

    all_quarters: set[str] = set()
    raw_series: dict[str, pd.Series] = {}
    for code in postal_codes:
        norm = _normalize_code(code)
        series = quarterly_area_prices(
            work_df, norm, building_type, price_column=price_column
        )
        raw_series[norm] = series
        all_quarters.update(series.index.astype(str))

    quarters = sorted(all_quarters, key=quarter_index)
    out_series: list[tuple[str, tuple[float | None, ...]]] = []
    for code in [_normalize_code(c) for c in postal_codes]:
        series = raw_series.get(code, pd.Series(dtype=float))
        if index_to_100:
            series = index_series_to_100(series)
        values: list[float | None] = []
        for q in quarters:
            value = series.get(q, float("nan"))
            if value is None or pd.isna(value):
                values.append(None)
            else:
                values.append(float(value))
        out_series.append((code, tuple(values)))

    return CompareChartData(
        quarters=tuple(quarters),
        series=tuple(out_series),
        indexed=index_to_100,
        use_real=use_real,
    )


def build_compare_figure(data: CompareChartData) -> go.Figure:
    fig = go.Figure()
    x = list(data.quarters)
    if data.indexed:
        y_title = "Index (100 = start)"
    elif data.use_real:
        y_title = "Real EUR/m²"
    else:
        y_title = "EUR/m²"

    for label, values in data.series:
        y = [None if v is None else float(v) for v in values]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=label,
                connectgaps=False,
            )
        )

    fig.update_layout(
        yaxis_title=y_title,
        xaxis_title="Quarter",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        height=380,
    )
    return fig


def summaries_for_compare(
    prices_df: pd.DataFrame,
    quarter: str,
    building_type: str | None,
    *,
    cpi_quarterly: pd.Series | None = None,
) -> pd.DataFrame:
    df_real = None
    if cpi_quarterly is not None and not cpi_quarterly.empty:
        df_real = to_real(prices_df, cpi_quarterly)
    return summarize_areas(
        prices_df, quarter, building_type=building_type, df_real=df_real
    )
