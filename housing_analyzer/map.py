"""Map layer metrics and Plotly choropleth figures (no Streamlit)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from housing_analyzer.analysis.metrics import (
    _quarter_index,
    _shift_quarter,
    summarize_area,
)
from housing_analyzer.data.boundaries import join_prices_to_areas

MISSING_COLOR = "#bdbdbd"
METRIC_PRICE = "price_per_sqm"
METRIC_CHANGE_1Y = "pct_change_1y"
METRIC_CHANGE_5Y = "pct_change_5y"
METRIC_SALES = "sales_4q"

METRIC_CHOICES: tuple[tuple[str, str], ...] = (
    (METRIC_PRICE, "Price per square metre"),
    (METRIC_CHANGE_1Y, "1-year change"),
    (METRIC_CHANGE_5Y, "5-year change"),
    (METRIC_SALES, "Number of sales (last 4 quarters)"),
)

BUILDING_TYPE_CHOICES: tuple[tuple[str, str], ...] = (
    ("all", "All building types"),
    ("1", "One-room flats"),
    ("2", "Two-room flats"),
    ("3", "Three-plus-room flats"),
    ("5", "Terraced houses"),
)

METRIC_UNITS: dict[str, str] = {
    METRIC_PRICE: "EUR/m²",
    METRIC_CHANGE_1Y: "%",
    METRIC_CHANGE_5Y: "%",
    METRIC_SALES: "sales",
}

NO_DATA_HOVER = "No data (too few sales or not published)"
LOW_RELIABILITY_HOVER = "Based on few sales"


def _filter_building_type(df: pd.DataFrame, building_type_code: str | None) -> pd.DataFrame:
    if building_type_code is None or building_type_code == "all":
        return df
    code = str(building_type_code)
    return df.loc[
        df["building_type"].eq(code)
        | df["building_type"].str.startswith(f"{code} —")
        | df["building_type"].str.startswith(f"{code} -")
    ]


def resolve_building_type_label(
    prices_df: pd.DataFrame, building_type_code: str | None
) -> str | None:
    if building_type_code is None or building_type_code == "all":
        return None
    filtered = _filter_building_type(prices_df, building_type_code)
    if filtered.empty:
        return None
    return str(filtered["building_type"].iloc[0])


def latest_quarter_with_data(
    prices_df: pd.DataFrame, building_type_code: str | None = None
) -> str | None:
    """Return the latest quarter that has at least one published price."""
    filtered = _filter_building_type(prices_df, building_type_code)
    if filtered.empty:
        return None
    priced = filtered.loc[filtered["price_per_sqm"].notna()]
    if priced.empty:
        return None
    quarters = sorted(priced["quarter"].unique(), key=_quarter_index)
    return quarters[-1]


def list_quarters(prices_df: pd.DataFrame) -> list[str]:
    if prices_df.empty:
        return []
    return sorted(prices_df["quarter"].unique(), key=_quarter_index)


def trailing_sales_sum(
    prices_df: pd.DataFrame,
    postal_code: str,
    quarter: str,
    building_type_code: str | None,
    *,
    window: int = 4,
) -> float:
    area = prices_df.loc[prices_df["postal_code"] == str(postal_code).zfill(5)]
    area = _filter_building_type(area, building_type_code)
    if area.empty:
        return float("nan")
    total = 0
    seen_any = False
    for offset in range(window):
        q = _shift_quarter(quarter, offset)
        qrows = area.loc[area["quarter"] == q]
        if qrows.empty:
            continue
        txs = qrows["transactions"].dropna()
        if not txs.empty:
            seen_any = True
            total += int(txs.sum())
    if not seen_any:
        return float("nan")
    return float(total)


def _metric_raw_value(row: Mapping[str, Any], metric: str) -> float:
    key = metric
    if metric not in row:
        return float("nan")
    value = row[key]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if pd.isna(value):
        return float("nan")
    return float(value)


def metric_is_missing(row: Mapping[str, Any], metric: str) -> bool:
    return np.isnan(_metric_raw_value(row, metric))


def metric_color_range(values: pd.Series, metric: str) -> tuple[float, float]:
    valid = values.dropna()
    if valid.empty:
        return 0.0, 1.0
    if metric in (METRIC_CHANGE_1Y, METRIC_CHANGE_5Y):
        bound = max(float(valid.abs().max()), 1.0)
        return -bound, bound
    low = float(valid.min())
    high = float(valid.max())
    if low == high:
        return low, low + 1.0
    return low, high


def format_metric_value(value: float, metric: str) -> str:
    if np.isnan(value):
        return "—"
    unit = METRIC_UNITS[metric]
    if metric == METRIC_PRICE:
        return f"{value:,.0f} {unit}"
    if metric in (METRIC_CHANGE_1Y, METRIC_CHANGE_5Y):
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f} {unit}"
    if metric == METRIC_SALES:
        return f"{int(round(value))} {unit}"
    return f"{value} {unit}"


def reliability_display(label: str | None) -> str:
    if label is None:
        return "Unknown"
    mapping = {
        "ok": "OK — enough sales",
        "low": "Low — based on few sales",
        "none": "No published price",
        "unknown": "Unknown (pre-2020)",
    }
    return mapping.get(str(label), str(label))


def format_hover_text(row: Mapping[str, Any], metric: str) -> str:
    postal = str(row.get("postal_code", "")).zfill(5)
    name = row.get("area_name") or ""
    if metric_is_missing(row, metric):
        return (
            f"<b>{postal}</b> {name}<br>"
            f"{NO_DATA_HOVER}<br>"
            f"Reliability: {reliability_display(row.get('reliability'))}"
        )

    metric_label = dict(METRIC_CHOICES).get(metric, metric)
    value_text = format_metric_value(_metric_raw_value(row, metric), metric)
    sales = row.get("sales_4q")
    sales_text = "—"
    if sales is not None and not (isinstance(sales, float) and np.isnan(sales)) and not pd.isna(sales):
        sales_text = str(int(round(float(sales))))

    lines = [
        f"<b>{postal}</b> {name}",
        f"{metric_label}: {value_text}",
        f"Sales (last 4 quarters): {sales_text}",
        f"Reliability: {reliability_display(row.get('reliability'))}",
    ]
    if str(row.get("reliability")) == "low":
        lines.append(LOW_RELIABILITY_HOVER)
    return "<br>".join(lines)


def prepare_map_dataframe(
    prices_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    quarter: str,
    building_type_code: str | None,
    metric: str,
) -> pd.DataFrame:
    """Join boundaries to metrics for one quarter, building type, and map layer."""
    code = None if building_type_code in (None, "all") else building_type_code
    join = join_prices_to_areas(prices_df, boundaries, quarter, code)
    frame = join.frame.copy()
    bt_label = resolve_building_type_label(prices_df, code)

    enriched_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        postal = str(row["postal_code"]).zfill(5)
        if not row.get("has_boundary", True):
            continue
        summary = summarize_area(
            prices_df, postal, quarter, building_type=bt_label
        )
        sales = trailing_sales_sum(prices_df, postal, quarter, code)
        enriched = {
            "postal_code": postal,
            "area_name": row["area_name"],
            "price_per_sqm": summary["price_per_sqm"],
            "pct_change_1y": summary["pct_change_1y"],
            "pct_change_5y": summary["pct_change_5y"],
            "sales_4q": sales,
            "reliability": summary["reliability"],
            "transactions_quarter": row["transactions"],
        }
        enriched["missing"] = metric_is_missing(enriched, metric)
        enriched["hover"] = format_hover_text(enriched, metric)
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows)


def _colorscale_with_opacity(base: str, opacity: float) -> list[list[Any]]:
    """Return a two-stop colorscale for Plotly (values 0..1)."""
    return [[0.0, base], [1.0, base]]


def build_choropleth_figure(
    map_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    metric: str,
    *,
    color_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Build a MapLibre choropleth with grey areas for missing metric values."""
    if map_df.empty:
        fig = go.Figure()
        fig.update_layout(
            map_style="carto-positron",
            map_center={"lat": 64.5, "lon": 26.0},
            map_zoom=4,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        return fig

    metric_title = dict(METRIC_CHOICES).get(metric, metric)
    unit = METRIC_UNITS[metric]
    color_label = f"{metric_title} ({unit})"

    with_data = map_df.loc[~map_df["missing"]].copy()
    missing = map_df.loc[map_df["missing"]].copy()

    zmin, zmax = color_range or metric_color_range(
        map_df.loc[~map_df["missing"], metric], metric
    )

    fig = go.Figure()

    if not with_data.empty:
        low = with_data["reliability"] == "low"
        line_width = np.where(low, 2.0, 0.5)
        line_color = np.where(low, "#616161", "white")
        fig.add_trace(
            go.Choroplethmap(
                geojson=boundaries,
                locations=with_data["postal_code"],
                z=with_data[metric],
                featureidkey="properties.postal_code",
                colorscale="Viridis",
                zmin=zmin,
                zmax=zmax,
                marker={
                    "line": {
                        "width": line_width.tolist(),
                        "color": line_color.tolist(),
                    }
                },
                showscale=True,
                colorbar={"title": color_label},
                customdata=with_data[["postal_code", "area_name"]],
                hovertext=with_data["hover"],
                hoverinfo="text",
                name="Areas with data",
            )
        )

    if not missing.empty:
        fig.add_trace(
            go.Choroplethmap(
                geojson=boundaries,
                locations=missing["postal_code"],
                z=[0.0] * len(missing),
                featureidkey="properties.postal_code",
                colorscale=_colorscale_with_opacity(MISSING_COLOR, 1.0),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={"line": {"width": 0.5, "color": "#757575"}},
                hovertext=missing["hover"],
                hoverinfo="text",
                name="No data",
            )
        )

    fig.update_layout(
        map_style="carto-positron",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig
