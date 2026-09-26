"""Market activity: trailing sales per 1,000 inhabitants by postal area."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MIN_POPULATION = 100
DEFAULT_LOW_RELIABILITY_MIN_SALES = 10

MARKET_ACTIVITY_LOW_RELIABILITY_CAPTION = (
    "Based on fewer than ten sales in the last four quarters; "
    "the activity rate may be unstable."
)


def market_activity(
    sales_4q: float,
    population: float,
    *,
    min_population: int = DEFAULT_MIN_POPULATION,
) -> float:
    """Sales in the last four quarters per 1,000 inhabitants.

    Returns NaN when sales or population are missing, or when population is below
    ``min_population``. A true zero sale count yields 0.0, not missing.
    """
    if min_population < 1:
        raise ValueError("min_population must be at least 1")
    if sales_4q is None or population is None:
        return float("nan")
    if (isinstance(sales_4q, float) and np.isnan(sales_4q)) or pd.isna(sales_4q):
        return float("nan")
    if (isinstance(population, float) and np.isnan(population)) or pd.isna(population):
        return float("nan")
    pop = float(population)
    if pop < float(min_population):
        return float("nan")
    return float(sales_4q) * 1000.0 / pop


def market_activity_reliability_note(
    sales_4q: float,
    *,
    min_sales: int = DEFAULT_LOW_RELIABILITY_MIN_SALES,
) -> str | None:
    """Caption when trailing sales are under the usual low-reliability threshold."""
    if min_sales < 1:
        raise ValueError("min_sales must be at least 1")
    if sales_4q is None or (isinstance(sales_4q, float) and np.isnan(sales_4q)) or pd.isna(
        sales_4q
    ):
        return None
    if float(sales_4q) < float(min_sales):
        return MARKET_ACTIVITY_LOW_RELIABILITY_CAPTION
    return None


def market_activity_table(
    sales_by_area: pd.Series,
    demographics_df: pd.DataFrame,
    *,
    min_population: int = DEFAULT_MIN_POPULATION,
) -> pd.DataFrame:
    """Per-area sales, population, and activity rate aligned to ``sales_by_area`` index."""
    demo = demographics_df.copy()
    demo["postal_code"] = demo["postal_code"].astype(str).str.zfill(5)
    demo_index = demo.set_index("postal_code")

    codes = sales_by_area.index.astype(str).str.zfill(5)
    populations = demo_index.reindex(codes)["population"].to_numpy(dtype=float)
    sales_vals = sales_by_area.to_numpy(dtype=float)
    activity = np.array(
        [
            market_activity(s, p, min_population=min_population)
            for s, p in zip(sales_vals, populations, strict=True)
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "postal_code": codes.to_numpy(),
            "sales_4q": sales_vals,
            "population": populations,
            "market_activity": activity,
        }
    )
