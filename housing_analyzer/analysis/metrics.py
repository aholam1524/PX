"""Pure analysis metrics on tidy housing price tables."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

_KEY_COLS = ("postal_code", "building_type", "quarter")
_PRE_2020_QUARTER = "2020Q1"


def _quarter_index(quarter: str) -> int:
    year = int(quarter[:4])
    q = int(quarter[5])
    return year * 4 + (q - 1)


def _quarter_from_index(index: int) -> str:
    year = index // 4
    q = index % 4 + 1
    return f"{year}Q{q}"


def _shift_quarter(quarter: str, quarters_back: int) -> str:
    return _quarter_from_index(_quarter_index(quarter) - quarters_back)


def _is_pre_2020(quarter: str) -> bool:
    return _quarter_index(quarter) < _quarter_index(_PRE_2020_QUARTER)


def _aggregate_area_price(
    rows: pd.DataFrame,
) -> tuple[float, str | None]:
    """One postal code in one quarter: weighted or simple mean price."""
    prices = rows["price_per_sqm"].astype(float)
    if prices.notna().sum() == 0:
        return float("nan"), None

    tx = rows["transactions"]
    has_tx = tx.notna() & (tx.astype(float) > 0)
    weighted_rows = rows.loc[has_tx & prices.notna()]
    if not weighted_rows.empty:
        weights = weighted_rows["transactions"].astype(float)
        value = float(
            np.average(weighted_rows["price_per_sqm"].astype(float), weights=weights)
        )
        return value, "weighted_mean"

    valid = prices.dropna()
    return float(valid.mean()), "simple_mean"


def _weighted_or_simple_mean(
    values: pd.Series, weights: pd.Series
) -> tuple[float, str]:
    valid = values.notna()
    if not valid.any():
        return float("nan"), "simple_mean"

    w = weights.reindex(values.index).astype(float)
    mask = valid & w.notna() & (w > 0)
    if mask.any():
        return float(np.average(values[mask], weights=w[mask])), "weighted_mean"

    return float(values[valid].mean()), "simple_mean"


def pct_change(df: pd.DataFrame, quarters: int) -> pd.Series:
    """Percentage change in ``price_per_sqm`` versus ``quarters`` calendar quarters earlier."""
    if quarters < 1:
        raise ValueError("quarters must be at least 1")

    lookup = df.set_index(list(_KEY_COLS))["price_per_sqm"]
    result = pd.Series(index=df.index, dtype=float)

    for idx, row in df.iterrows():
        current = row["price_per_sqm"]
        prior_quarter = _shift_quarter(row["quarter"], quarters)
        key = (row["postal_code"], row["building_type"], prior_quarter)
        try:
            prior = lookup.loc[key]
        except KeyError:
            prior = float("nan")
        if pd.isna(current) or pd.isna(prior):
            result.loc[idx] = float("nan")
        else:
            result.loc[idx] = (float(current) / float(prior) - 1.0) * 100.0

    return result


def reliability(
    df: pd.DataFrame,
    *,
    min_transactions: int = 10,
    window: int = 4,
) -> pd.Series:
    """Reliability label per row from trailing transaction counts and price presence."""
    if min_transactions < 1:
        raise ValueError("min_transactions must be at least 1")
    if window < 1:
        raise ValueError("window must be at least 1")

    tx_lookup = df.set_index(list(_KEY_COLS))["transactions"]
    labels = pd.Series(index=df.index, dtype=object)

    for idx, row in df.iterrows():
        quarter = row["quarter"]
        if _is_pre_2020(quarter):
            labels.loc[idx] = "unknown"
            continue
        if pd.isna(row["price_per_sqm"]):
            labels.loc[idx] = "none"
            continue

        total = 0
        for offset in range(window):
            q = _shift_quarter(quarter, offset)
            key = (row["postal_code"], row["building_type"], q)
            try:
                tx = tx_lookup.loc[key]
            except KeyError:
                continue
            if pd.notna(tx):
                total += int(tx)

        labels.loc[idx] = "ok" if total >= min_transactions else "low"

    return labels


def rank_percentile(
    df: pd.DataFrame,
    quarter: str,
    building_type: str | None = None,
) -> pd.DataFrame:
    """Rank (1 = highest price) and percentile for one quarter across postal-code areas."""
    subset = df.loc[df["quarter"] == quarter].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "postal_code",
                "area_name",
                "price_per_sqm",
                "price_method",
                "rank",
                "percentile",
            ]
        )

    if building_type is not None:
        subset = subset.loc[subset["building_type"] == building_type]

    area_rows: list[dict[str, Any]] = []
    for postal_code, group in subset.groupby("postal_code", sort=False):
        area_name = group["area_name"].iloc[0]
        if building_type is not None:
            price = float(group["price_per_sqm"].iloc[0])
            price_method = None if pd.isna(price) else "observed"
        else:
            price, price_method = _aggregate_area_price(group)

        area_rows.append(
            {
                "postal_code": postal_code,
                "area_name": area_name,
                "price_per_sqm": price,
                "price_method": price_method,
                "rank": pd.NA,
                "percentile": float("nan"),
            }
        )

    out = pd.DataFrame(area_rows)
    priced = out.loc[out["price_per_sqm"].notna()].copy()
    if priced.empty:
        return out

    n = len(priced)
    ranked = priced["price_per_sqm"].rank(method="min", ascending=False)
    priced["rank"] = ranked.astype("Int64")
    if n == 1:
        priced["percentile"] = 100.0
    else:
        priced["percentile"] = 100.0 * (1.0 - (ranked - 1.0) / (n - 1.0))

    out.loc[priced.index, ["rank", "percentile"]] = priced[["rank", "percentile"]]
    return out.sort_values("postal_code", ignore_index=True)


def regional_average(
    df: pd.DataFrame,
    group: Mapping[str, str],
    quarter: str,
    *,
    building_type: str | None = None,
) -> pd.DataFrame:
    """Transaction-weighted (or simple) mean ``price_per_sqm`` per named group of areas."""
    subset = df.loc[df["quarter"] == quarter].copy()
    if building_type is not None:
        subset = subset.loc[subset["building_type"] == building_type]

    subset = subset.loc[subset["postal_code"].isin(group)]
    if subset.empty:
        return pd.DataFrame(columns=["group", "price_per_sqm", "average_method"])

    subset = subset.assign(group=subset["postal_code"].map(group))
    subset = subset.dropna(subset=["group"])

    rows: list[dict[str, Any]] = []
    for group_name, chunk in subset.groupby("group", sort=True):
        value, method = _weighted_or_simple_mean(
            chunk["price_per_sqm"], chunk["transactions"]
        )
        rows.append(
            {
                "group": group_name,
                "price_per_sqm": value,
                "average_method": method,
            }
        )

    return pd.DataFrame(rows)


def _consolidate_reliability(labels: pd.Series) -> str | None:
    if labels.empty:
        return None
    values = list(labels.astype(str))
    if all(v == "unknown" for v in values):
        return "unknown"
    if any(v == "none" for v in values):
        return "none"
    if any(v == "low" for v in values):
        return "low"
    return "ok"


def _area_price_series(area_df: pd.DataFrame) -> dict[str, float]:
    series: dict[str, float] = {}
    for q in sorted(area_df["quarter"].unique(), key=_quarter_index):
        price, _ = _aggregate_area_price(area_df.loc[area_df["quarter"] == q])
        if pd.notna(price):
            series[q] = float(price)
    return series


def summarize_area(
    df: pd.DataFrame,
    postal_code: str,
    quarter: str,
    building_type: str | None = None,
) -> dict[str, Any]:
    """Latest price, YoY and 5y change, reliability, rank and percentile for one area."""
    postal_code = str(postal_code).zfill(5)
    area_df = df.loc[df["postal_code"] == postal_code]
    ranks = rank_percentile(df, quarter, building_type=building_type)
    rank_row = ranks.loc[ranks["postal_code"] == postal_code]

    if area_df.empty:
        return {
            "postal_code": postal_code,
            "quarter": quarter,
            "building_type": building_type,
            "price_per_sqm": float("nan"),
            "pct_change_1y": float("nan"),
            "pct_change_5y": float("nan"),
            "reliability": None,
            "rank": pd.NA,
            "percentile": float("nan"),
            "price_method": None,
        }

    if building_type is not None:
        type_df = area_df.loc[area_df["building_type"] == building_type]
        row = type_df.loc[type_df["quarter"] == quarter]
        price = row["price_per_sqm"].iloc[0] if len(row) else float("nan")
        price_method = "observed" if pd.notna(price) else None
        rel_labels = reliability(type_df)
        rel_row = type_df.loc[type_df["quarter"] == quarter]
        reliability_label = (
            rel_labels.loc[rel_row.index[0]] if len(rel_row) else None
        )
        changes = pct_change(type_df, 4), pct_change(type_df, 20)
        pct_1y = changes[0].loc[rel_row.index[0]] if len(rel_row) else float("nan")
        pct_5y = changes[1].loc[rel_row.index[0]] if len(rel_row) else float("nan")
    else:
        if len(rank_row):
            price = rank_row["price_per_sqm"].iloc[0]
            price_method = rank_row["price_method"].iloc[0]
        else:
            price, price_method = _aggregate_area_price(
                area_df.loc[area_df["quarter"] == quarter]
            )
        rel_at_q = area_df.loc[area_df["quarter"] == quarter]
        reliability_label = _consolidate_reliability(reliability(area_df).loc[rel_at_q.index])
        series = _area_price_series(area_df)
        pct_1y = _pct_from_series(series, quarter, 4)
        pct_5y = _pct_from_series(series, quarter, 20)

    rank_val = rank_row["rank"].iloc[0] if len(rank_row) else pd.NA
    percentile_val = rank_row["percentile"].iloc[0] if len(rank_row) else float("nan")

    return {
        "postal_code": postal_code,
        "quarter": quarter,
        "building_type": building_type,
        "price_per_sqm": float(price) if pd.notna(price) else float("nan"),
        "pct_change_1y": float(pct_1y) if pd.notna(pct_1y) else float("nan"),
        "pct_change_5y": float(pct_5y) if pd.notna(pct_5y) else float("nan"),
        "reliability": reliability_label,
        "rank": rank_val,
        "percentile": float(percentile_val) if pd.notna(percentile_val) else float("nan"),
        "price_method": price_method,
    }


def _pct_from_series(series: dict[str, float], quarter: str, lag: int) -> float:
    if quarter not in series:
        return float("nan")
    prior_q = _shift_quarter(quarter, lag)
    current = series.get(quarter)
    prior = series.get(prior_q)
    if current is None or prior is None or pd.isna(current) or pd.isna(prior):
        return float("nan")
    return (current / prior - 1.0) * 100.0
