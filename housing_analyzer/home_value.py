"""Estimate home value from area price movement since purchase (not a valuation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from housing_analyzer.analysis.metrics import (
    area_prices_at,
    quarter_index,
    reliability_at,
)
from housing_analyzer.data.cpi import latest_complete_quarter
from housing_analyzer.map import latest_quarter_with_data, list_quarters
from housing_analyzer.panel import index_series_to_100, quarterly_area_prices

HOME_VALUE_DISCLAIMER = (
    "This is not a valuation. It applies the average price movement of the area to your "
    "purchase price; the real value of your home depends on its condition and features, "
    "and small areas with few sales can be noisy."
)

_PRE_2020 = quarter_index("2020Q1")


@dataclass(frozen=True)
class HomeValueEstimate:
    """Result of :func:`estimate_home_value`."""

    enough_data: bool
    postal_code: str
    building_type: str | None
    purchase_quarter_requested: str
    purchase_quarter_used: str | None
    purchase_price_per_sqm: float | None
    latest_quarter_requested: str | None
    latest_quarter_used: str | None
    latest_price_per_sqm: float | None
    purchase_price: float
    price_index: float | None
    estimated_value: float | None
    nominal_change_pct: float | None
    real_change_pct: float | None
    purchase_sales_count: float | None
    latest_sales_count: float | None
    reliability: str | None
    notes: list[str] = field(default_factory=list)


def _area_exists(prices_df: pd.DataFrame, postal_code: str) -> bool:
    code = str(postal_code).zfill(5)
    codes = prices_df["postal_code"].astype(str).str.zfill(5)
    return bool(codes.eq(code).any())


def _price_at_quarter(
    prices_df: pd.DataFrame,
    postal_code: str,
    quarter: str,
    building_type: str | None,
) -> float | None:
    code = str(postal_code).zfill(5)
    prices = area_prices_at(prices_df, quarter, building_type)
    if code not in prices.index:
        return None
    value = float(prices.loc[code, "price_per_sqm"])
    if pd.isna(value):
        return None
    return value


def _transactions_at_quarter(
    prices_df: pd.DataFrame,
    postal_code: str,
    quarter: str,
    building_type: str | None,
) -> float | None:
    if quarter_index(quarter) < _PRE_2020:
        return None
    code = str(postal_code).zfill(5)
    area = prices_df.loc[prices_df["postal_code"].astype(str).str.zfill(5) == code]
    if building_type is not None:
        area = area.loc[area["building_type"] == building_type]
    rows = area.loc[area["quarter"] == quarter, "transactions"]
    if rows.empty:
        return None
    total = rows.dropna().astype(float).sum(min_count=1)
    if pd.isna(total):
        return None
    return float(total)


def nearest_quarter_with_price(
    prices_df: pd.DataFrame,
    postal_code: str,
    target_quarter: str,
    building_type: str | None,
    *,
    max_gap: int = 2,
) -> tuple[str | None, float | None]:
    """Nearest quarter within ``max_gap`` that has a published area price."""
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    target_idx = quarter_index(target_quarter)
    best: tuple[int, str, float] | None = None
    for q in list_quarters(prices_df):
        dist = abs(quarter_index(q) - target_idx)
        if dist > max_gap:
            continue
        price = _price_at_quarter(prices_df, postal_code, q, building_type)
        if price is None:
            continue
        if best is None or dist < best[0] or (dist == best[0] and quarter_index(q) < quarter_index(best[1])):
            best = (dist, q, price)
    if best is None:
        return None, None
    return best[1], best[2]


def _real_change_pct(
    purchase_price: float,
    estimated_value: float,
    cpi_quarterly: pd.Series,
    purchase_quarter: str,
    latest_quarter: str,
) -> float | None:
    if cpi_quarterly.empty:
        return None
    base = latest_complete_quarter(cpi_quarterly) or latest_quarter
    for q in (purchase_quarter, latest_quarter, base):
        if q not in cpi_quarterly.index or pd.isna(cpi_quarterly.loc[q]):
            return None
    cpi_p = float(cpi_quarterly.loc[purchase_quarter])
    cpi_l = float(cpi_quarterly.loc[latest_quarter])
    cpi_b = float(cpi_quarterly.loc[base])
    if cpi_p == 0.0 or cpi_l == 0.0:
        return None
    real_purchase = purchase_price * (cpi_b / cpi_p)
    real_estimated = estimated_value * (cpi_b / cpi_l)
    if real_purchase == 0.0:
        return None
    return (real_estimated / real_purchase - 1.0) * 100.0


def estimate_home_value(
    prices_df: pd.DataFrame,
    cpi_quarterly: pd.Series,
    postal_code: str,
    building_type: str | None,
    purchase_quarter: str,
    purchase_price: float,
    *,
    latest_quarter: str | None = None,
    max_gap: int = 2,
) -> HomeValueEstimate:
    """Illustrative value from area price index since purchase (not a valuation)."""
    code = str(postal_code).zfill(5)
    notes: list[str] = []
    latest = latest_quarter or latest_quarter_with_data(
        prices_df,
        None if building_type is None else _building_type_code(building_type),
    )

    base = HomeValueEstimate(
        enough_data=False,
        postal_code=code,
        building_type=building_type,
        purchase_quarter_requested=purchase_quarter,
        purchase_quarter_used=None,
        purchase_price_per_sqm=None,
        latest_quarter_requested=latest,
        latest_quarter_used=None,
        latest_price_per_sqm=None,
        purchase_price=purchase_price,
        price_index=None,
        estimated_value=None,
        nominal_change_pct=None,
        real_change_pct=None,
        purchase_sales_count=None,
        latest_sales_count=None,
        reliability=None,
        notes=notes,
    )

    if purchase_price <= 0:
        notes.append("Purchase price must be greater than zero.")
        return base

    if not _area_exists(prices_df, code):
        notes.append(f"No price data for postal code {code}.")
        return base

    if latest is None:
        notes.append("No quarters with price data are available.")
        return base

    if quarter_index(purchase_quarter) > quarter_index(latest):
        notes.append(
            f"Purchase quarter {purchase_quarter} is after the latest data quarter {latest}."
        )
        return base

    purchase_q_used, purchase_ppsm = nearest_quarter_with_price(
        prices_df, code, purchase_quarter, building_type, max_gap=max_gap
    )
    latest_q_used, latest_ppsm = nearest_quarter_with_price(
        prices_df, code, latest, building_type, max_gap=max_gap
    )

    if purchase_q_used is None:
        notes.append(
            f"No published area price within {max_gap} quarters of purchase quarter "
            f"{purchase_quarter}."
        )
        return base
    if latest_q_used is None:
        notes.append(
            f"No published area price within {max_gap} quarters of latest quarter {latest}."
        )
        return base

    if purchase_q_used != purchase_quarter:
        notes.append(
            f"Purchase price level taken from {purchase_q_used}, the nearest quarter with data."
        )
    if latest_q_used != latest:
        notes.append(
            f"Latest price level taken from {latest_q_used}, the nearest quarter with data."
        )

    if purchase_ppsm is None or latest_ppsm is None or purchase_ppsm == 0.0:
        notes.append("Not enough price data to compute an area index.")
        return base

    index = latest_ppsm / purchase_ppsm
    estimated = purchase_price * index
    nominal_pct = (index - 1.0) * 100.0
    real_pct = _real_change_pct(
        purchase_price, estimated, cpi_quarterly, purchase_q_used, latest_q_used
    )
    if real_pct is None:
        notes.append("Inflation-adjusted change is unavailable (missing CPI for some quarters).")

    rel = reliability_at(prices_df, latest_q_used, building_type)
    reliability_label = str(rel.loc[code]) if code in rel.index else None

    return HomeValueEstimate(
        enough_data=True,
        postal_code=code,
        building_type=building_type,
        purchase_quarter_requested=purchase_quarter,
        purchase_quarter_used=purchase_q_used,
        purchase_price_per_sqm=purchase_ppsm,
        latest_quarter_requested=latest,
        latest_quarter_used=latest_q_used,
        latest_price_per_sqm=latest_ppsm,
        purchase_price=purchase_price,
        price_index=index,
        estimated_value=estimated,
        nominal_change_pct=nominal_pct,
        real_change_pct=real_pct,
        purchase_sales_count=_transactions_at_quarter(
            prices_df, code, purchase_q_used, building_type
        ),
        latest_sales_count=_transactions_at_quarter(
            prices_df, code, latest_q_used, building_type
        ),
        reliability=reliability_label,
        notes=notes,
    )


def _building_type_code(building_type_label: str) -> str | None:
    """Best-effort code from a label like ``1 — flat`` for quarter helpers."""
    text = str(building_type_label).strip()
    if not text:
        return None
    return text.split()[0] if text[0].isdigit() else None


def indexed_area_chart_from_purchase(
    prices_df: pd.DataFrame,
    postal_code: str,
    building_type: str | None,
    purchase_quarter_used: str,
    *,
    through_quarter: str | None = None,
) -> go.Figure:
    """Area EUR/m² indexed to 100 at ``purchase_quarter_used`` (gaps stay gaps)."""
    code = str(postal_code).zfill(5)
    series = quarterly_area_prices(prices_df, code, building_type)
    end = through_quarter or (series.dropna().index[-1] if not series.dropna().empty else purchase_quarter_used)
    from_idx = quarter_index(purchase_quarter_used)
    end_idx = quarter_index(end)
    quarters = [q for q in series.index if from_idx <= quarter_index(q) <= end_idx]
    trimmed = series.reindex(quarters)
    base = trimmed.get(purchase_quarter_used)
    if base is not None and pd.notna(base) and float(base) != 0.0:
        indexed = (trimmed.astype(float) / float(base)) * 100.0
    else:
        indexed = index_series_to_100(trimmed)
    fig = go.Figure(
        go.Scatter(
            x=list(indexed.index),
            y=[None if pd.isna(v) else float(v) for v in indexed.to_numpy()],
            mode="lines+markers",
            name="Area price index",
            connectgaps=False,
        )
    )
    fig.update_layout(
        yaxis_title="Index (100 = purchase quarter)",
        xaxis_title="Quarter",
        height=320,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    return fig


def estimate_as_dict(result: HomeValueEstimate) -> dict[str, Any]:
    """JSON-friendly view of a :class:`HomeValueEstimate`."""
    return {
        "enough_data": result.enough_data,
        "postal_code": result.postal_code,
        "building_type": result.building_type,
        "purchase_quarter_requested": result.purchase_quarter_requested,
        "purchase_quarter_used": result.purchase_quarter_used,
        "purchase_price_per_sqm": result.purchase_price_per_sqm,
        "latest_quarter_requested": result.latest_quarter_requested,
        "latest_quarter_used": result.latest_quarter_used,
        "latest_price_per_sqm": result.latest_price_per_sqm,
        "purchase_price": result.purchase_price,
        "price_index": result.price_index,
        "estimated_value": result.estimated_value,
        "nominal_change_pct": result.nominal_change_pct,
        "real_change_pct": result.real_change_pct,
        "purchase_sales_count": result.purchase_sales_count,
        "latest_sales_count": result.latest_sales_count,
        "reliability": result.reliability,
        "notes": list(result.notes),
    }
