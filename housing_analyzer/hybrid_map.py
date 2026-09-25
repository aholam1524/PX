"""Hybrid postal + municipality map layers (pure helpers, no Streamlit)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from housing_analyzer.affordability import (
    BUDGET_FIT_OVER,
    BUDGET_FIT_STRETCH,
    BUDGET_FIT_WITHIN,
    classify_budget_fit_ratio,
    price_to_budget_ratio,
    typical_dwelling_price,
)
from housing_analyzer.data.municipalities import (
    municipality_code_from_properties,
    municipality_metrics,
    municipality_real_changes,
    municipality_year_for_quarter,
)
from housing_analyzer.data import paths as data_paths
from housing_analyzer.map import (
    BUDGET_FIT_COLORS,
    BUDGET_FIT_LABELS,
    METRIC_CHANGE_1Y,
    METRIC_CHANGE_1Y_REAL,
    METRIC_CHANGE_5Y,
    METRIC_CHANGE_5Y_REAL,
    METRIC_CHOICES,
    METRIC_FITS_BUDGET,
    METRIC_PRICE,
    METRIC_PRICE_TO_INCOME,
    METRIC_SALES,
    METRIC_UNITS,
    MISSING_COLOR,
    MISSING_OUTLINE_COLOR,
    MISSING_OUTLINE_WIDTH,
    NO_DATA_FILL,
    VALUE_COLORSCALE,
    MAP_LAYOUT_MARGINS,
    _PCT_CHANGE_METRICS,
    _REAL_CHANGE_METRICS,
    _budget_ratio_and_category,
    _feature_subset,
    _reliability_outlines,
    _solid_colorscale,
    build_choropleth_figure,
    value_colorbar,
    format_metric_value,
    metric_color_range,
    metric_is_missing,
    postal_code_from_selection,
    reliability_display,
)

MUNICIPALITY_BUILDING_ALL = "1"
MUNICIPALITY_BUILDING_FLATS = "3"
MUNICIPALITY_BUILDING_TERRACED = "4"

MUNICIPALITY_TRACE_NAME = "Municipality averages"
POSTAL_TRACE_NAME = "Areas with data"
POSTAL_FALLBACK_TRACE_NAME = "Postal areas (municipality value)"
GREY_TRACE_NAME = "No data"

TRANSPARENT_FILL = "rgba(0,0,0,0)"
MUNICIPALITY_FILL_OPACITY = 0.55

HYBRID_METRICS = frozenset(
    {
        METRIC_PRICE,
        METRIC_CHANGE_1Y,
        METRIC_CHANGE_5Y,
        METRIC_CHANGE_1Y_REAL,
        METRIC_CHANGE_5Y_REAL,
        METRIC_FITS_BUDGET,
    }
)

_HYBRID_UNSUPPORTED_REASONS: dict[str, str] = {
    METRIC_SALES: (
        "Municipality sales are reported as annual totals, not as a trailing "
        "four-quarter count, so the municipality layer is not shown for this metric."
    ),
    METRIC_PRICE_TO_INCOME: (
        "Price-to-income uses Paavo income at postal-code level; municipality "
        "figures are not drawn for this layer."
    ),
}


@dataclass(frozen=True)
class MapSelection:
    """Map click: postal code plus whether the value comes from postal or municipality data."""

    postal_code: str
    level: str  # "postal" | "municipality"


@dataclass(frozen=True)
class MunicipalityMapCard:
    """Detail-panel content for a municipality-backed map selection."""

    municipality_code: str
    municipality_name: str
    year: int
    price_per_sqm: float | None
    pct_change_1y: float | None
    pct_change_5y: float | None
    transactions: int | None
    reliability: str | None


@dataclass(frozen=True)
class HybridCoverageCounts:
    own_postal: int
    municipality_coloured: int
    no_data: int


def municipality_files_available() -> bool:
    """True when committed snapshot or fixture municipality files can be loaded."""
    if data_paths.use_fixtures():
        return (
            data_paths.MUNICIPALITY_PRICES_FIXTURE_FILE.is_file()
            and data_paths.MUNICIPALITY_BOUNDARIES_FIXTURE_FILE.is_file()
        )
    return (
        data_paths.MUNICIPALITY_PRICES_SNAPSHOT_FILE.is_file()
        and data_paths.MUNICIPALITY_BOUNDARIES_SNAPSHOT_FILE.is_file()
    )


def hybrid_metric_supported(metric: str) -> bool:
    return metric in HYBRID_METRICS


def hybrid_unsupported_reason(metric: str) -> str | None:
    return _HYBRID_UNSUPPORTED_REASONS.get(metric)


def municipality_building_type_for_map(
    building_type_code: str | None,
) -> tuple[str | None, bool]:
    """Map sidebar building-type codes to municipality table codes.

    Returns ``(municipality_code, approximate)`` where *approximate* is True when
    room-specific flat types are folded into the generic flats category.
    """
    if building_type_code in (None, "all"):
        return MUNICIPALITY_BUILDING_ALL, False
    if building_type_code in ("1", "2", "3"):
        return MUNICIPALITY_BUILDING_FLATS, True
    if building_type_code == "5":
        return MUNICIPALITY_BUILDING_TERRACED, False
    return MUNICIPALITY_BUILDING_ALL, True


def building_type_mapping_caption(building_type_code: str | None) -> str | None:
    """Short caption when municipality building types differ from the map selection."""
    _code, approximate = municipality_building_type_for_map(building_type_code)
    if not approximate:
        return None
    return (
        "Municipality values use the Statistics Finland flats category (all room "
        "counts combined), not separate one-, two-, or three-plus-room series."
    )


def postal_to_municipality_codes(boundaries: Mapping[str, Any]) -> dict[str, str]:
    """Postal code (five digits) -> municipality code (three digits)."""
    mapping: dict[str, str] = {}
    for feature in boundaries.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        postal = str(props.get("postal_code") or props.get("posti_alue", "")).zfill(5)
        if not postal.strip("0"):
            continue
        muni = municipality_code_from_properties(props)
        if muni is not None:
            mapping[postal] = muni
    return mapping


def _municipality_feature_subset(
    boundaries: Mapping[str, Any], codes: Any
) -> dict[str, Any]:
    wanted = {str(code).zfill(3) for code in codes}
    features = [
        feature
        for feature in boundaries.get("features") or []
        if str((feature.get("properties") or {}).get("municipality_code", "")).zfill(3)
        in wanted
    ]
    return {"type": "FeatureCollection", "features": features}


def _municipality_metric_frame(
    mun_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    year: int,
    building_type_code: str | None,
    metric: str,
    *,
    size_sqm: float = 0.0,
    max_affordable_price: float = 0.0,
) -> pd.DataFrame:
    mun_bt, _ = municipality_building_type_for_map(building_type_code)
    if metric in _REAL_CHANGE_METRICS:
        base = municipality_real_changes(mun_df, cpi_df, year, mun_bt)
    else:
        base = municipality_metrics(mun_df, year, mun_bt)

    if base.empty:
        return base

    records: list[dict[str, Any]] = []
    for row in base.to_dict("records"):
        record = dict(row)
        if metric == METRIC_FITS_BUDGET:
            price = row.get("price_per_sqm")
            if price is None or (isinstance(price, float) and np.isnan(price)) or pd.isna(
                price
            ):
                record["budget_fit"] = None
                record["budget_ratio"] = float("nan")
                record["missing"] = True
            else:
                typical = typical_dwelling_price(float(price), size_sqm)
                ratio, category = _budget_ratio_and_category(typical, max_affordable_price)
                record["budget_fit"] = category
                record["budget_ratio"] = ratio
                record["missing"] = False
            record[metric] = record.get("budget_ratio", float("nan"))
        else:
            record["missing"] = metric_is_missing(record, metric)
        records.append(record)

    return pd.DataFrame(records)


def format_municipality_hover_text(
    row: Mapping[str, Any],
    metric: str,
    *,
    year: int,
    cpi_available: bool = True,
    size_sqm: float = 0.0,
    max_affordable_price: float = 0.0,
) -> str:
    name = row.get("municipality_name") or row.get("municipality_code") or ""
    header = f"Municipality {name} - annual figure {year}"
    if metric == METRIC_FITS_BUDGET:
        if row.get("missing"):
            return f"{header}<br>No data for this building type and year."
        price_sqm = row.get("price_per_sqm")
        typical = typical_dwelling_price(float(price_sqm), size_sqm)
        ratio, category = _budget_ratio_and_category(typical, max_affordable_price)
        ratio_display = "∞" if math.isinf(ratio) else f"{ratio:.2f}"
        lines = [
            header,
            f"Typical price ({size_sqm:g} m²): {typical:,.0f} EUR",
            f"Max affordable: {max_affordable_price:,.0f} EUR",
            f"Ratio to budget: {ratio_display}",
            f"Fits budget: {BUDGET_FIT_LABELS[category]}",
            f"Reliability: {reliability_display(row.get('reliability'))}",
            "Municipality average; the postal-code area has no published price",
        ]
        return "<br>".join(lines)

    if metric_is_missing(row, metric):
        reason = (
            "No data (CPI not final for this year yet)"
            if metric in _REAL_CHANGE_METRICS and not cpi_available
            else "No data (too few sales or not published)"
        )
        return f"{header}<br>{reason}"

    metric_label = dict(METRIC_CHOICES).get(metric, metric)
    value_text = format_metric_value(_metric_raw_value(row, metric), metric)
    sales = row.get("transactions")
    sales_text = "—"
    if sales is not None and not pd.isna(sales):
        sales_text = str(int(sales))

    lines = [
        header,
        f"{metric_label}: {value_text}",
        f"Sales ({year}): {sales_text}",
        f"Reliability: {reliability_display(row.get('reliability'))}",
        "Municipality average; the postal-code area has no published price",
    ]
    if str(row.get("reliability")) == "low":
        lines.append("Based on few sales")
    return "<br>".join(lines)


def _metric_raw_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == METRIC_FITS_BUDGET:
        value = row.get("budget_ratio")
    else:
        value = row.get(metric)
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return float("nan")
    return float(value)


def prepare_municipality_map_dataframe(
    mun_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
    metric: str,
    *,
    size_sqm: float = 0.0,
    max_affordable_price: float = 0.0,
) -> tuple[pd.DataFrame, int]:
    """Municipality metrics for one map layer; returns frame and calendar year used."""
    years = sorted({int(y) for y in mun_df["year"].unique()}) if not mun_df.empty else []
    choice = municipality_year_for_quarter(quarter, years)
    frame = _municipality_metric_frame(
        mun_df,
        cpi_df,
        choice.year,
        building_type_code,
        metric,
        size_sqm=size_sqm,
        max_affordable_price=max_affordable_price,
    )
    if frame.empty:
        return frame, choice.year

    from housing_analyzer.data.municipalities import cpi_by_year

    yearly_cpi = cpi_by_year(cpi_df)
    cpi_available = (
        metric not in _REAL_CHANGE_METRICS
        or (choice.year in yearly_cpi.index and not pd.isna(yearly_cpi.loc[choice.year]))
    )

    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        record = dict(row)
        record["hover"] = format_municipality_hover_text(
            record,
            metric,
            year=choice.year,
            cpi_available=cpi_available,
            size_sqm=size_sqm,
            max_affordable_price=max_affordable_price,
        )
        records.append(record)
    return pd.DataFrame(records), choice.year


def classify_hybrid_postal_coverage(
    map_df: pd.DataFrame,
    metric: str,
    postal_to_muni: Mapping[str, str],
    municipality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Label each postal row as postal, municipality fallback, or no data."""
    if map_df.empty:
        return map_df.copy()

    if municipality_df.empty:
        out = map_df.copy()
        out["coverage"] = np.where(map_df["missing"], "none", "postal")
        return out

    mun_index = municipality_df.set_index("municipality_code")
    coverage: list[str] = []
    for row in map_df.to_dict("records"):
        if not row.get("missing"):
            coverage.append("postal")
            continue
        muni = postal_to_muni.get(str(row["postal_code"]).zfill(5))
        if muni is None or muni not in mun_index.index:
            coverage.append("none")
            continue
        mun_row = mun_index.loc[muni]
        if metric == METRIC_FITS_BUDGET:
            has_value = not bool(mun_row.get("missing", True))
        else:
            has_value = not metric_is_missing(mun_row, metric)
        coverage.append("municipality" if has_value else "none")

    out = map_df.copy()
    out["coverage"] = coverage
    return out


def hybrid_coverage_counts(map_df: pd.DataFrame) -> HybridCoverageCounts:
    if map_df.empty or "coverage" not in map_df.columns:
        n = len(map_df)
        own = int((~map_df["missing"]).sum()) if not map_df.empty else 0
        return HybridCoverageCounts(own_postal=own, municipality_coloured=0, no_data=n - own)
    series = map_df["coverage"]
    return HybridCoverageCounts(
        own_postal=int((series == "postal").sum()),
        municipality_coloured=int((series == "municipality").sum()),
        no_data=int((series == "none").sum()),
    )


def hybrid_metric_color_range(
    postal_df: pd.DataFrame,
    municipality_df: pd.DataFrame,
    metric: str,
    *,
    use_full_range: bool = False,
) -> tuple[float, float]:
    """Shared colour scale over postal and municipality values."""
    parts: list[pd.Series] = []
    if not postal_df.empty and "coverage" in postal_df.columns:
        parts.append(postal_df.loc[postal_df["coverage"] == "postal", metric])
    elif not postal_df.empty:
        parts.append(postal_df.loc[~postal_df["missing"], metric])
    if not municipality_df.empty:
        parts.append(municipality_df.loc[~municipality_df["missing"], metric])
    if not parts:
        return metric_color_range(pd.Series(dtype=float), metric)
    combined = pd.concat(parts, ignore_index=True)
    return metric_color_range(combined, metric, use_full_range=use_full_range)


def municipality_map_card_data(
    mun_df: pd.DataFrame,
    municipality_code: str,
    quarter: str,
    building_type_code: str | None,
) -> MunicipalityMapCard | None:
    years = sorted({int(y) for y in mun_df["year"].unique()}) if not mun_df.empty else []
    if not years:
        return None
    choice = municipality_year_for_quarter(quarter, years)
    mun_bt, _ = municipality_building_type_for_map(building_type_code)
    metrics = municipality_metrics(mun_df, choice.year, mun_bt)
    code = str(municipality_code).zfill(3)
    row = metrics.loc[metrics["municipality_code"] == code]
    if row.empty:
        return None
    item = row.iloc[0]
    tx = item.get("transactions")
    tx_val = None if pd.isna(tx) else int(tx)
    return MunicipalityMapCard(
        municipality_code=code,
        municipality_name=str(item["municipality_name"]),
        year=choice.year,
        price_per_sqm=float(item["price_per_sqm"])
        if pd.notna(item["price_per_sqm"])
        else None,
        pct_change_1y=float(item["pct_change_1y"])
        if pd.notna(item["pct_change_1y"])
        else None,
        pct_change_5y=float(item["pct_change_5y"])
        if pd.notna(item["pct_change_5y"])
        else None,
        transactions=tx_val,
        reliability=str(item["reliability"]) if item.get("reliability") else None,
    )


def map_selection_from_event(selection: Any) -> MapSelection | None:
    """Postal code and data level from a Streamlit Plotly selection event."""
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
        level = "postal"
        if custom and len(custom) > 2 and custom[2]:
            level = str(custom[2])
        elif point.get("curveNumber") is not None:
            # Fallback when customdata lacks level: trace order is fixed in hybrid figures.
            pass
        code = None
        if custom:
            code = str(custom[0]).zfill(5)
        if code is None:
            location = point.get("location")
            if location:
                code = str(location).zfill(5)
        if code:
            return MapSelection(postal_code=code, level=level)
    return None


def map_selection_from_event_or_postal(selection: Any) -> MapSelection | None:
    """Prefer :func:`map_selection_from_event`, else legacy postal-only helper."""
    picked = map_selection_from_event(selection)
    if picked is not None:
        return picked
    code = postal_code_from_selection(selection)
    if code:
        return MapSelection(postal_code=code, level="postal")
    return None


def _municipality_hover_by_postal(
    postal_codes: pd.Series,
    postal_to_muni: Mapping[str, str],
    municipality_df: pd.DataFrame,
    fallback_hover: pd.Series | None = None,
) -> list[str]:
    mun_index = municipality_df.set_index("municipality_code")
    hovers: list[str] = []
    for postal in postal_codes:
        code = str(postal).zfill(5)
        muni = postal_to_muni.get(code)
        if muni is not None and muni in mun_index.index:
            hovers.append(str(mun_index.loc[muni]["hover"]))
        elif fallback_hover is not None and code in fallback_hover.index:
            hovers.append(str(fallback_hover.loc[code]))
        else:
            hovers.append("")
    return hovers


def _add_postal_customdata(frame: pd.DataFrame, level: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in frame.to_dict("records"):
        rows.append(
            [
                str(row["postal_code"]).zfill(5),
                row.get("area_name") or "",
                level,
            ]
        )
    return rows


def build_hybrid_choropleth_figure(
    postal_df: pd.DataFrame,
    municipality_df: pd.DataFrame,
    postal_boundaries: Mapping[str, Any],
    municipality_boundaries: Mapping[str, Any],
    metric: str,
    *,
    color_range: tuple[float, float] | None = None,
    fill_gaps: bool = True,
) -> go.Figure:
    """Choropleth with optional municipality layer underneath postal detail."""
    if not fill_gaps or not hybrid_metric_supported(metric):
        return build_choropleth_figure(
            postal_df, postal_boundaries, metric, color_range=color_range
        )

    if metric == METRIC_FITS_BUDGET:
        return _build_hybrid_budget_figure(
            postal_df,
            municipality_df,
            postal_boundaries,
            municipality_boundaries,
        )

    if postal_df.empty and municipality_df.empty:
        return build_choropleth_figure(postal_df, postal_boundaries, metric)

    metric_title = dict(METRIC_CHOICES).get(metric, metric)
    unit = METRIC_UNITS[metric]
    color_label = f"{metric_title} ({unit})"

    if "coverage" not in postal_df.columns:
        postal_df = classify_hybrid_postal_coverage(
            postal_df,
            metric,
            postal_to_municipality_codes(postal_boundaries),
            municipality_df,
        )

    with_data = postal_df.loc[postal_df["coverage"] == "postal"].copy()
    fallback = postal_df.loc[postal_df["coverage"] == "municipality"].copy()
    missing = postal_df.loc[postal_df["coverage"] == "none"].copy()
    if municipality_df.empty or "missing" not in municipality_df.columns:
        mun_with = municipality_df.iloc[0:0]
    else:
        mun_with = municipality_df.loc[~municipality_df["missing"]].copy()

    zmin, zmax = color_range or hybrid_metric_color_range(
        postal_df, municipality_df, metric
    )

    fig = go.Figure()

    if not mun_with.empty:
        low = mun_with["reliability"] == "low"
        line_width = np.where(low, 1.5, 0.3)
        line_color = np.where(low, "#9e9e9e", "#eeeeee")
        fig.add_trace(
            go.Choroplethmap(
                geojson=_municipality_feature_subset(
                    municipality_boundaries, mun_with["municipality_code"]
                ),
                locations=mun_with["municipality_code"],
                z=mun_with[metric],
                featureidkey="properties.municipality_code",
                colorscale=VALUE_COLORSCALE,
                zmin=zmin,
                zmax=zmax,
                marker={
                    "line": {"width": line_width.tolist(), "color": line_color.tolist()},
                    "opacity": MUNICIPALITY_FILL_OPACITY,
                },
                showscale=False,
                hovertext=mun_with["hover"],
                hoverinfo="text",
                name=MUNICIPALITY_TRACE_NAME,
            )
        )

    if not with_data.empty:
        outlines = _reliability_outlines(with_data["reliability"])
        fig.add_trace(
            go.Choroplethmap(
                geojson=_feature_subset(postal_boundaries, with_data["postal_code"]),
                locations=with_data["postal_code"],
                z=with_data[metric],
                featureidkey="properties.postal_code",
                colorscale=VALUE_COLORSCALE,
                zmin=zmin,
                zmax=zmax,
                marker={"line": outlines},
                showscale=True,
                colorbar=value_colorbar(metric, color_label, zmin, zmax),
                customdata=_add_postal_customdata(with_data, "postal"),
                hovertext=with_data["hover"],
                hoverinfo="text",
                name=POSTAL_TRACE_NAME,
            )
        )

    p2m = postal_to_municipality_codes(postal_boundaries)

    if not fallback.empty:
        hover = _municipality_hover_by_postal(
            fallback["postal_code"], p2m, municipality_df, fallback["hover"]
        )
        fig.add_trace(
            go.Choroplethmap(
                geojson=_feature_subset(postal_boundaries, fallback["postal_code"]),
                locations=fallback["postal_code"],
                z=[1.0] * len(fallback),
                featureidkey="properties.postal_code",
                colorscale=_solid_colorscale(TRANSPARENT_FILL),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={"line": {"width": 0.2, "color": "rgba(255,255,255,0.3)"}},
                customdata=_add_postal_customdata(fallback, "municipality"),
                hovertext=hover,
                hoverinfo="text",
                name=POSTAL_FALLBACK_TRACE_NAME,
            )
        )

    if not missing.empty:
        fig.add_trace(
            go.Choroplethmap(
                geojson=_feature_subset(postal_boundaries, missing["postal_code"]),
                locations=missing["postal_code"],
                z=[0.0] * len(missing),
                featureidkey="properties.postal_code",
                colorscale=_solid_colorscale(NO_DATA_FILL),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={
                    "line": {
                        "width": MISSING_OUTLINE_WIDTH,
                        "color": MISSING_OUTLINE_COLOR,
                    }
                },
                customdata=_add_postal_customdata(missing, "none"),
                hovertext=missing["hover"],
                hoverinfo="text",
                name=GREY_TRACE_NAME,
            )
        )

    fig.update_layout(
        map_style="carto-positron",
        margin=MAP_LAYOUT_MARGINS,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def _build_hybrid_budget_figure(
    postal_df: pd.DataFrame,
    municipality_df: pd.DataFrame,
    postal_boundaries: Mapping[str, Any],
    municipality_boundaries: Mapping[str, Any],
) -> go.Figure:
    if "coverage" not in postal_df.columns:
        postal_df = classify_hybrid_postal_coverage(
            postal_df,
            METRIC_FITS_BUDGET,
            postal_to_municipality_codes(postal_boundaries),
            municipality_df,
        )

    fig = go.Figure()
    order = (BUDGET_FIT_WITHIN, BUDGET_FIT_STRETCH, BUDGET_FIT_OVER)

    for category in order:
        subset = municipality_df.loc[municipality_df["budget_fit"] == category]
        if subset.empty:
            continue
        color = BUDGET_FIT_COLORS[category]
        fig.add_trace(
            go.Choroplethmap(
                geojson=_municipality_feature_subset(
                    municipality_boundaries, subset["municipality_code"]
                ),
                locations=subset["municipality_code"],
                z=[1.0] * len(subset),
                featureidkey="properties.municipality_code",
                colorscale=_solid_colorscale(color),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={
                    "line": {"width": 0.3, "color": "#eeeeee"},
                    "opacity": MUNICIPALITY_FILL_OPACITY,
                },
                hovertext=subset["hover"],
                hoverinfo="text",
                name=f"{BUDGET_FIT_LABELS[category]} (municipality)",
            )
        )

    for category in order:
        subset = postal_df.loc[
            (postal_df["coverage"] == "postal") & (postal_df["budget_fit"] == category)
        ]
        if subset.empty:
            continue
        color = BUDGET_FIT_COLORS[category]
        fig.add_trace(
            go.Choroplethmap(
                geojson=_feature_subset(postal_boundaries, subset["postal_code"]),
                locations=subset["postal_code"],
                z=[1.0] * len(subset),
                featureidkey="properties.postal_code",
                colorscale=_solid_colorscale(color),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={"line": {"width": 0.5, "color": "white"}},
                customdata=_add_postal_customdata(subset, "postal"),
                hovertext=subset["hover"],
                hoverinfo="text",
                name=BUDGET_FIT_LABELS[category],
            )
        )

    fallback = postal_df.loc[postal_df["coverage"] == "municipality"]
    if not fallback.empty:
        p2m = postal_to_municipality_codes(postal_boundaries)
        hover = _municipality_hover_by_postal(
            fallback["postal_code"], p2m, municipality_df, fallback["hover"]
        )
        fig.add_trace(
            go.Choroplethmap(
                geojson=_feature_subset(postal_boundaries, fallback["postal_code"]),
                locations=fallback["postal_code"],
                z=[1.0] * len(fallback),
                featureidkey="properties.postal_code",
                colorscale=_solid_colorscale(TRANSPARENT_FILL),
                zmin=0,
                zmax=1,
                showscale=False,
                marker={"line": {"width": 0.2, "color": "rgba(255,255,255,0.3)"}},
                customdata=_add_postal_customdata(fallback, "municipality"),
                hovertext=hover,
                hoverinfo="text",
                name=POSTAL_FALLBACK_TRACE_NAME,
            )
        )

    missing = postal_df.loc[postal_df["coverage"] == "none"]
    if not missing.empty:
        fig.add_trace(
            go.Choroplethmap(
                geojson=_feature_subset(postal_boundaries, missing["postal_code"]),
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
                customdata=_add_postal_customdata(missing, "none"),
                hovertext=missing["hover"],
                hoverinfo="text",
                name=GREY_TRACE_NAME,
            )
        )

    fig.update_layout(
        map_style="carto-positron",
        margin=MAP_LAYOUT_MARGINS,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig
