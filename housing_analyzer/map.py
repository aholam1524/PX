"""Map layer metrics and Plotly choropleth figures (no Streamlit)."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from housing_analyzer.analysis.metrics import (
    quarter_index,
    shift_quarter,
    summarize_areas,
    to_real,
)
from housing_analyzer.analysis.relationships import price_to_income_ratio
from housing_analyzer.data.boundaries import join_prices_to_areas
from housing_analyzer.data.cpi import cpi_by_quarter, load_cpi
from housing_analyzer.data.demographics import load_demographics
from housing_analyzer.affordability import (
    BUDGET_FIT_OVER,
    BUDGET_FIT_STRETCH,
    BUDGET_FIT_WITHIN,
    classify_budget_fit_ratio,
    price_to_budget_ratio,
    typical_dwelling_price,
)

MISSING_COLOR = "#e5e7eb"
MISSING_OUTLINE_COLOR = "#d1d5db"
MISSING_OUTLINE_WIDTH = 0.3
PROVISIONAL_COVERAGE_RATIO = 0.85
PRIOR_QUARTERS_FOR_COVERAGE = 8
DEFAULT_COLOR_PERCENTILE_LOW = 2.0
DEFAULT_COLOR_PERCENTILE_HIGH = 98.0
BUDGET_FIT_COLOR_WITHIN = "#2e7d32"
BUDGET_FIT_COLOR_STRETCH = "#f9a825"
BUDGET_FIT_COLOR_OVER = "#c62828"
METRIC_PRICE = "price_per_sqm"
METRIC_CHANGE_1Y = "pct_change_1y"
METRIC_CHANGE_5Y = "pct_change_5y"
METRIC_CHANGE_1Y_REAL = "pct_change_1y_real"
METRIC_CHANGE_5Y_REAL = "pct_change_5y_real"
METRIC_SALES = "sales_4q"
METRIC_PRICE_TO_INCOME = "price_to_income"
METRIC_FITS_BUDGET = "fits_budget"

METRIC_CHOICES: tuple[tuple[str, str], ...] = (
    (METRIC_PRICE, "Price per square metre"),
    (METRIC_CHANGE_1Y, "1-year change"),
    (METRIC_CHANGE_5Y, "5-year change"),
    (METRIC_CHANGE_1Y_REAL, "1-year change (real)"),
    (METRIC_CHANGE_5Y_REAL, "5-year change (real)"),
    (METRIC_SALES, "Number of sales (last 4 quarters)"),
    (METRIC_PRICE_TO_INCOME, "Price-to-income ratio (rough)"),
    (METRIC_FITS_BUDGET, "Fits my budget"),
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
    METRIC_CHANGE_1Y_REAL: "%",
    METRIC_CHANGE_5Y_REAL: "%",
    METRIC_SALES: "sales",
    METRIC_PRICE_TO_INCOME: "years income / m²",
    METRIC_FITS_BUDGET: "vs budget",
}

BUDGET_FIT_LABELS: dict[str, str] = {
    BUDGET_FIT_WITHIN: "Within budget",
    BUDGET_FIT_STRETCH: "Up to 20% over budget",
    BUDGET_FIT_OVER: "More than 20% over budget",
}

BUDGET_FIT_COLORS: dict[str, str] = {
    BUDGET_FIT_WITHIN: BUDGET_FIT_COLOR_WITHIN,
    BUDGET_FIT_STRETCH: BUDGET_FIT_COLOR_STRETCH,
    BUDGET_FIT_OVER: BUDGET_FIT_COLOR_OVER,
}

_PCT_CHANGE_METRICS = frozenset(
    {
        METRIC_CHANGE_1Y,
        METRIC_CHANGE_5Y,
        METRIC_CHANGE_1Y_REAL,
        METRIC_CHANGE_5Y_REAL,
    }
)

_REAL_CHANGE_METRICS = frozenset({METRIC_CHANGE_1Y_REAL, METRIC_CHANGE_5Y_REAL})

NO_DATA_HOVER = "No data (too few sales or not published)"
NO_CPI_HOVER = "No data (CPI not final for this quarter yet)"
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
    quarters = sorted(priced["quarter"].unique(), key=quarter_index)
    return quarters[-1]


def count_areas_with_published_price(
    prices_df: pd.DataFrame, quarter: str, building_type_code: str | None = None
) -> int:
    """Postal codes with a published price in ``quarter`` (after building-type filter)."""
    filtered = _filter_building_type(prices_df, building_type_code)
    if filtered.empty:
        return 0
    rows = filtered.loc[
        (filtered["quarter"] == quarter) & filtered["price_per_sqm"].notna()
    ]
    return int(rows["postal_code"].nunique())


def _median_prior_quarter_coverage(
    prices_df: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
    *,
    prior_quarters: int = PRIOR_QUARTERS_FOR_COVERAGE,
) -> float | None:
    filtered = _filter_building_type(prices_df, building_type_code)
    if filtered.empty:
        return None
    quarters = sorted(filtered["quarter"].unique(), key=quarter_index)
    if quarter not in quarters:
        return None
    idx = quarters.index(quarter)
    prior = quarters[max(0, idx - prior_quarters) : idx]
    if not prior:
        return None
    counts = [
        count_areas_with_published_price(prices_df, q, building_type_code) for q in prior
    ]
    return float(np.median(counts))


def typical_quarter_price_coverage(
    prices_df: pd.DataFrame,
    quarter: str,
    building_type_code: str | None = None,
    *,
    prior_quarters: int = PRIOR_QUARTERS_FOR_COVERAGE,
) -> float | None:
    """Median number of areas with a published price over the prior ``prior_quarters``."""
    return _median_prior_quarter_coverage(
        prices_df, quarter, building_type_code, prior_quarters=prior_quarters
    )


def quarter_meets_coverage_threshold(
    prices_df: pd.DataFrame,
    quarter: str,
    building_type_code: str | None = None,
    *,
    ratio: float = PROVISIONAL_COVERAGE_RATIO,
    prior_quarters: int = PRIOR_QUARTERS_FOR_COVERAGE,
) -> bool:
    """True when ``quarter`` has at least ``ratio`` of the median prior-quarter coverage."""
    typical = _median_prior_quarter_coverage(
        prices_df, quarter, building_type_code, prior_quarters=prior_quarters
    )
    if typical is None or typical <= 0:
        return True
    count = count_areas_with_published_price(prices_df, quarter, building_type_code)
    return count >= ratio * typical


def default_map_quarter(
    prices_df: pd.DataFrame, building_type_code: str | None = None
) -> str | None:
    """Latest quarter with enough published-price coverage; else latest with any price."""
    filtered = _filter_building_type(prices_df, building_type_code)
    if filtered.empty:
        return None
    quarters = sorted(filtered["quarter"].unique(), key=quarter_index)
    for quarter in reversed(quarters):
        if quarter_meets_coverage_threshold(prices_df, quarter, building_type_code):
            return quarter
    return latest_quarter_with_data(prices_df, building_type_code)


def list_quarters(prices_df: pd.DataFrame) -> list[str]:
    if prices_df.empty:
        return []
    return sorted(prices_df["quarter"].unique(), key=quarter_index)


def trailing_sales_by_area(
    prices_df: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
    *,
    window: int = 4,
) -> pd.Series:
    """Sales over the trailing ``window`` quarters for every postal code at once.

    Areas with no reported transaction count in the window are NaN (not zero).
    """
    quarters = {shift_quarter(quarter, offset) for offset in range(window)}
    rows = _filter_building_type(
        prices_df.loc[prices_df["quarter"].isin(quarters)], building_type_code
    )
    return (
        rows["transactions"]
        .astype(float)
        .groupby(rows["postal_code"])
        .sum(min_count=1)
    )


def trailing_sales_sum(
    prices_df: pd.DataFrame,
    postal_code: str,
    quarter: str,
    building_type_code: str | None,
    *,
    window: int = 4,
) -> float:
    totals = trailing_sales_by_area(
        prices_df, quarter, building_type_code, window=window
    )
    return float(totals.get(str(postal_code).zfill(5), float("nan")))


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


def metric_color_range(
    values: pd.Series,
    metric: str,
    *,
    percentile_low: float = DEFAULT_COLOR_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_COLOR_PERCENTILE_HIGH,
    use_full_range: bool = False,
) -> tuple[float, float]:
    valid = values.dropna()
    if valid.empty:
        return 0.0, 1.0
    clip = not use_full_range and not (
        percentile_low <= 0 and percentile_high >= 100
    )
    if metric in _PCT_CHANGE_METRICS:
        if clip:
            bound = max(float(np.percentile(valid.abs(), percentile_high)), 1.0)
        else:
            bound = max(float(valid.abs().max()), 1.0)
        return -bound, bound
    if clip:
        low = float(np.percentile(valid, percentile_low))
        high = float(np.percentile(valid, percentile_high))
    else:
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
    if metric in _PCT_CHANGE_METRICS:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f} {unit}"
    if metric == METRIC_SALES:
        return f"{int(round(value))} {unit}"
    if metric == METRIC_PRICE_TO_INCOME:
        return f"{value:.2f} {unit}"
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


def format_hover_text(
    row: Mapping[str, Any], metric: str, *, cpi_available: bool = True
) -> str:
    postal = str(row.get("postal_code", "")).zfill(5)
    name = row.get("area_name") or ""
    if metric_is_missing(row, metric):
        reason = (
            NO_CPI_HOVER
            if metric in _REAL_CHANGE_METRICS and not cpi_available
            else NO_DATA_HOVER
        )
        return (
            f"<b>{postal}</b> {name}<br>"
            f"{reason}<br>"
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
    *,
    cpi_df: pd.DataFrame | None = None,
    demographics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join boundaries to metrics for one quarter, building type, and map layer.

    All areas are computed in one pass over the data (not one call per area), so
    the cost does not grow with the number of postal-code areas squared.
    """
    code = None if building_type_code in (None, "all") else building_type_code
    join = join_prices_to_areas(prices_df, boundaries, quarter, code)
    frame = join.frame
    if not frame.empty and "has_boundary" in frame.columns:
        frame = frame.loc[frame["has_boundary"]]
    if frame.empty:
        return pd.DataFrame()

    bt_label = resolve_building_type_label(prices_df, code)
    codes = frame["postal_code"].astype(str).str.zfill(5)
    cpi = cpi_df if cpi_df is not None else load_cpi()
    cpi_quarterly = cpi_by_quarter(cpi)
    cpi_available = quarter in cpi_quarterly.index
    prices_real = to_real(prices_df, cpi_quarterly)
    summaries = summarize_areas(
        prices_df, quarter, building_type=bt_label, df_real=prices_real
    ).reindex(codes)
    sales = trailing_sales_by_area(prices_df, quarter, code).reindex(codes)

    demo = demographics_df if demographics_df is not None else load_demographics()
    demo_index = demo.set_index(demo["postal_code"].astype(str).str.zfill(5))
    median_income = demo_index.reindex(codes)["median_income_eur"].to_numpy(dtype=float)
    price_vals = summaries["price_per_sqm"].to_numpy(dtype=float)
    price_to_income = np.array(
        [
            price_to_income_ratio(float(p), float(i))
            for p, i in zip(price_vals, median_income, strict=True)
        ],
        dtype=float,
    )

    enriched = pd.DataFrame(
        {
            "postal_code": codes.to_numpy(),
            "area_name": frame["area_name"].to_numpy(),
            "price_per_sqm": price_vals,
            "price_to_income": price_to_income,
            "pct_change_1y": summaries["pct_change_1y"].to_numpy(dtype=float),
            "pct_change_5y": summaries["pct_change_5y"].to_numpy(dtype=float),
            "pct_change_1y_real": summaries["pct_change_1y_real"].to_numpy(dtype=float),
            "pct_change_5y_real": summaries["pct_change_5y_real"].to_numpy(dtype=float),
            "sales_4q": sales.to_numpy(dtype=float),
            "reliability": [
                value if isinstance(value, str) else None
                for value in summaries["reliability"]
            ],
            "transactions_quarter": frame["transactions"].to_numpy(),
        }
    )
    records = enriched.to_dict("records")
    for record in records:
        record["missing"] = metric_is_missing(record, metric)
        record["hover"] = format_hover_text(
            record, metric, cpi_available=cpi_available
        )
    return pd.DataFrame(records)


def _budget_ratio_and_category(
    typical_price: float, max_affordable_price: float
) -> tuple[float, str]:
    """Ratio/category for one area, treating a non-positive budget as unaffordable
    rather than raising (a 0 EUR max affordable price is a valid user input, e.g.
    100% down payment)."""
    if max_affordable_price <= 0:
        return float("inf"), BUDGET_FIT_OVER
    ratio = price_to_budget_ratio(typical_price, max_affordable_price)
    return ratio, classify_budget_fit_ratio(ratio)


def format_budget_fit_hover(
    row: Mapping[str, Any],
    *,
    max_affordable_price: float,
    size_sqm: float,
) -> str:
    postal = str(row.get("postal_code", "")).zfill(5)
    name = row.get("area_name") or ""
    price_sqm = row.get("price_per_sqm")
    if price_sqm is None or (isinstance(price_sqm, float) and np.isnan(price_sqm)) or pd.isna(
        price_sqm
    ):
        return (
            f"<b>{postal}</b> {name}<br>"
            f"{NO_DATA_HOVER}<br>"
            f"Reliability: {reliability_display(row.get('reliability'))}"
        )
    typical = typical_dwelling_price(float(price_sqm), size_sqm)
    ratio, category = _budget_ratio_and_category(typical, max_affordable_price)
    ratio_display = "∞" if math.isinf(ratio) else f"{ratio:.2f}"
    lines = [
        f"<b>{postal}</b> {name}",
        f"Typical price ({size_sqm:g} m²): {typical:,.0f} EUR",
        f"Max affordable: {max_affordable_price:,.0f} EUR",
        f"Ratio to budget: {ratio_display}",
        f"Fits budget: {BUDGET_FIT_LABELS[category]}",
        f"Reliability: {reliability_display(row.get('reliability'))}",
    ]
    if str(row.get("reliability")) == "low":
        lines.append(LOW_RELIABILITY_HOVER)
    return "<br>".join(lines)


def prepare_budget_fit_dataframe(
    prices_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    quarter: str,
    building_type_code: str | None,
    size_sqm: float,
    max_affordable_price: float,
    *,
    cpi_df: pd.DataFrame | None = None,
    demographics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map layer comparing typical dwelling prices to a max affordable price."""
    base = prepare_map_dataframe(
        prices_df,
        boundaries,
        quarter,
        building_type_code,
        METRIC_PRICE,
        cpi_df=cpi_df,
        demographics_df=demographics_df,
    )
    if base.empty:
        return base

    records: list[dict[str, Any]] = []
    for row in base.to_dict("records"):
        price_sqm = row.get("price_per_sqm")
        missing_price = price_sqm is None or (
            isinstance(price_sqm, float) and np.isnan(price_sqm)
        ) or pd.isna(price_sqm)
        if missing_price:
            category = None
            ratio = float("nan")
            missing = True
        else:
            typical = typical_dwelling_price(float(price_sqm), size_sqm)
            ratio, category = _budget_ratio_and_category(typical, max_affordable_price)
            missing = False
        record = dict(row)
        record["budget_ratio"] = ratio
        record["budget_fit"] = category
        record["missing"] = missing
        record["hover"] = format_budget_fit_hover(
            row,
            max_affordable_price=max_affordable_price,
            size_sqm=size_sqm,
        )
        records.append(record)
    return pd.DataFrame(records)


def search_area_matches(map_df: pd.DataFrame, query: str) -> list[str]:
    """Postal codes whose code or area name contains ``query`` (case-insensitive)."""
    text = query.strip().lower()
    if not text or map_df.empty:
        return []
    codes = map_df["postal_code"].astype(str).str.zfill(5)
    names = map_df["area_name"].fillna("").astype(str).str.lower()
    mask = codes.str.contains(text, regex=False) | names.str.contains(text, regex=False)
    return sorted(set(codes[mask]))


def postal_code_from_selection(selection: Any) -> str | None:
    """Postal code of the first point in a Streamlit plotly selection event, if any."""
    if selection is None:
        return None
    payload = getattr(selection, "selection", None)
    if payload is None and isinstance(selection, Mapping):
        payload = selection.get("selection")
    if not payload or not hasattr(payload, "get"):
        return None
    for point in payload.get("points") or []:
        if not isinstance(point, Mapping):
            continue
        custom = point.get("customdata")
        if custom:
            return str(custom[0]).zfill(5)
        location = point.get("location")
        if location:
            return str(location).zfill(5)
    return None


def _solid_colorscale(color: str) -> list[list[Any]]:
    """Return a one-colour two-stop colorscale for Plotly (values 0..1)."""
    return [[0.0, color], [1.0, color]]


def _feature_subset(boundaries: Mapping[str, Any], codes: Any) -> dict[str, Any]:
    """GeoJSON with only the features for ``codes``, so each map trace carries its own areas.

    Passing the whole country to every trace roughly doubles the size of the figure that
    has to be built and sent to the browser on every interaction.
    """
    wanted = {str(code).zfill(5) for code in codes}
    features = [
        feature
        for feature in boundaries.get("features") or []
        if str((feature.get("properties") or {}).get("postal_code", "")).zfill(5) in wanted
    ]
    return {"type": "FeatureCollection", "features": features}


def build_budget_fit_choropleth_figure(
    map_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
) -> go.Figure:
    """Choropleth coloured by whether typical prices fit the user's budget."""
    if map_df.empty:
        fig = go.Figure()
        fig.update_layout(
            map_style="carto-positron",
            map_center={"lat": 64.5, "lon": 26.0},
            map_zoom=4,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        return fig

    fig = go.Figure()
    order = (BUDGET_FIT_WITHIN, BUDGET_FIT_STRETCH, BUDGET_FIT_OVER)
    for category in order:
        subset = map_df.loc[map_df["budget_fit"] == category]
        if subset.empty:
            continue
        color = BUDGET_FIT_COLORS[category]
        fig.add_trace(
            go.Choroplethmap(
                geojson=_feature_subset(boundaries, subset["postal_code"]),
                locations=subset["postal_code"],
                z=[1.0] * len(subset),
                featureidkey="properties.postal_code",
                colorscale=_solid_colorscale(color),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={"line": {"width": 0.5, "color": "white"}},
                hovertext=subset["hover"],
                hoverinfo="text",
                name=BUDGET_FIT_LABELS[category],
            )
        )

    missing = map_df.loc[map_df["missing"]]
    if not missing.empty:
        fig.add_trace(
            go.Choroplethmap(
                geojson=_feature_subset(boundaries, missing["postal_code"]),
                locations=missing["postal_code"],
                z=[0.0] * len(missing),
                featureidkey="properties.postal_code",
                colorscale=_solid_colorscale(MISSING_COLOR),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={
                    "line": {
                        "width": MISSING_OUTLINE_WIDTH,
                        "color": MISSING_OUTLINE_COLOR,
                    }
                },
                hovertext=missing["hover"],
                hoverinfo="text",
                name="No price data",
            )
        )

    fig.update_layout(
        map_style="carto-positron",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def build_choropleth_figure(
    map_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    metric: str,
    *,
    color_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Build a MapLibre choropleth with grey areas for missing metric values."""
    if metric == METRIC_FITS_BUDGET:
        return build_budget_fit_choropleth_figure(map_df, boundaries)

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
                geojson=_feature_subset(boundaries, with_data["postal_code"]),
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
                geojson=_feature_subset(boundaries, missing["postal_code"]),
                locations=missing["postal_code"],
                z=[0.0] * len(missing),
                featureidkey="properties.postal_code",
                colorscale=_solid_colorscale(MISSING_COLOR),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={
                    "line": {
                        "width": MISSING_OUTLINE_WIDTH,
                        "color": MISSING_OUTLINE_COLOR,
                    }
                },
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
