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


def quarter_index(quarter: str) -> int:
    """Public helper: a sortable integer for a quarter label like ``2025Q4``."""
    return _quarter_index(quarter)


def shift_quarter(quarter: str, quarters_back: int) -> str:
    """Public helper: the quarter label ``quarters_back`` quarters earlier."""
    return _shift_quarter(quarter, quarters_back)


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


_RANK_COLUMNS = [
    "postal_code",
    "area_name",
    "price_per_sqm",
    "price_method",
    "rank",
    "percentile",
]


def _quarter_slice(
    df: pd.DataFrame, quarter: str, building_type: str | None
) -> pd.DataFrame:
    subset = df.loc[df["quarter"] == quarter]
    if building_type is not None:
        subset = subset.loc[subset["building_type"] == building_type]
    return subset


def area_prices_at(
    df: pd.DataFrame,
    quarter: str,
    building_type: str | None = None,
) -> pd.DataFrame:
    """Price per square metre of every postal-code area for one quarter, all at once.

    Same rules as the per-area code: for one building type the observed price is
    used; without a building type the price is the transaction-weighted mean over
    building types (simple mean when no weights exist). Returns a frame indexed by
    ``postal_code`` with ``area_name``, ``price_per_sqm`` and ``price_method``.
    """
    columns = ["area_name", "price_per_sqm", "price_method"]
    subset = _quarter_slice(df, quarter, building_type)
    if subset.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="postal_code"))

    first = subset.drop_duplicates("postal_code", keep="first").set_index("postal_code")

    if building_type is not None:
        price = first["price_per_sqm"].astype(float)
        method = pd.Series(
            np.where(price.isna(), None, "observed"), index=price.index, dtype=object
        )
    else:
        postal = subset["postal_code"]
        prices = subset["price_per_sqm"].astype(float)
        tx = subset["transactions"].astype(float)
        weight = tx.where(tx > 0)
        usable = weight.notna() & prices.notna()
        numerator = (prices * weight).where(usable).groupby(postal).sum(min_count=1)
        denominator = weight.where(usable).groupby(postal).sum(min_count=1)
        weighted = numerator / denominator
        simple = prices.groupby(postal).mean()
        price = weighted.fillna(simple)
        method = pd.Series(
            np.where(
                weighted.notna(),
                "weighted_mean",
                np.where(simple.notna(), "simple_mean", None),
            ),
            index=weighted.index,
            dtype=object,
        )

    out = pd.DataFrame(
        {
            "area_name": first["area_name"],
            "price_per_sqm": price,
            "price_method": method,
        }
    )
    return out.sort_index()


def _add_rank_percentile(out: pd.DataFrame) -> pd.DataFrame:
    """Add ``rank`` (1 = highest price) and ``percentile`` columns to a price frame."""
    out = out.copy()
    out["rank"] = pd.NA
    out["percentile"] = float("nan")
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
    return out


def rank_percentile(
    df: pd.DataFrame,
    quarter: str,
    building_type: str | None = None,
) -> pd.DataFrame:
    """Rank (1 = highest price) and percentile for one quarter across postal-code areas."""
    prices = area_prices_at(df, quarter, building_type)
    if prices.empty:
        return pd.DataFrame(columns=_RANK_COLUMNS)

    out = _add_rank_percentile(prices)
    return out.reset_index()[_RANK_COLUMNS].sort_values("postal_code", ignore_index=True)


def reliability_at(
    df: pd.DataFrame,
    quarter: str,
    building_type: str | None = None,
    *,
    min_transactions: int = 10,
    window: int = 4,
) -> pd.Series:
    """Reliability label per postal code at one quarter, all areas at once.

    Same rules as :func:`reliability`. Without a building type the labels of the
    building types are combined (all unknown -> unknown, any none -> none, any
    low -> low, otherwise ok). Areas without a row at the quarter are absent.
    """
    if min_transactions < 1:
        raise ValueError("min_transactions must be at least 1")
    if window < 1:
        raise ValueError("window must be at least 1")

    subset = _quarter_slice(df, quarter, building_type)
    if subset.empty:
        return pd.Series(dtype=object, name="reliability")

    if _is_pre_2020(quarter):
        row_labels = pd.Series("unknown", index=subset.index, dtype=object)
    else:
        window_quarters = {_shift_quarter(quarter, offset) for offset in range(window)}
        win = df.loc[df["quarter"].isin(window_quarters)]
        if building_type is not None:
            win = win.loc[win["building_type"] == building_type]
        totals = (
            win["transactions"]
            .astype(float)
            .groupby([win["postal_code"], win["building_type"]])
            .sum()
        )
        keys = pd.MultiIndex.from_frame(subset[["postal_code", "building_type"]])
        row_totals = totals.reindex(keys).fillna(0.0).to_numpy()
        price_missing = subset["price_per_sqm"].isna().to_numpy()
        row_labels = pd.Series(
            np.where(
                price_missing,
                "none",
                np.where(row_totals >= min_transactions, "ok", "low"),
            ),
            index=subset.index,
            dtype=object,
        )

    postal = subset["postal_code"]
    if building_type is not None:
        labels = row_labels.groupby(postal).first()
    else:
        all_unknown = (row_labels == "unknown").groupby(postal).all()
        any_none = (row_labels == "none").groupby(postal).any()
        any_low = (row_labels == "low").groupby(postal).any()
        labels = pd.Series(
            np.where(
                all_unknown,
                "unknown",
                np.where(any_none, "none", np.where(any_low, "low", "ok")),
            ),
            index=all_unknown.index,
            dtype=object,
        )
    labels.name = "reliability"
    return labels.sort_index()


def summarize_areas(
    df: pd.DataFrame,
    quarter: str,
    building_type: str | None = None,
    *,
    min_transactions: int = 10,
    window: int = 4,
) -> pd.DataFrame:
    """:func:`summarize_area` for every postal-code area at once (one pass over the data).

    Returns a frame indexed by ``postal_code`` with ``price_per_sqm``,
    ``pct_change_1y``, ``pct_change_5y``, ``reliability``, ``rank``, ``percentile``
    and ``price_method``. Areas with no row at the quarter are absent.
    """
    columns = [
        "price_per_sqm",
        "pct_change_1y",
        "pct_change_5y",
        "reliability",
        "rank",
        "percentile",
        "price_method",
    ]
    current = area_prices_at(df, quarter, building_type)
    if current.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="postal_code"))

    ranked = _add_rank_percentile(current)
    now = current["price_per_sqm"]
    out = pd.DataFrame(index=current.index)
    out["price_per_sqm"] = now
    for name, lag in (("pct_change_1y", 4), ("pct_change_5y", 20)):
        prior = area_prices_at(df, _shift_quarter(quarter, lag), building_type)[
            "price_per_sqm"
        ].reindex(current.index)
        out[name] = (now / prior - 1.0) * 100.0
    out["reliability"] = reliability_at(
        df,
        quarter,
        building_type,
        min_transactions=min_transactions,
        window=window,
    ).reindex(current.index)
    out["rank"] = ranked["rank"]
    out["percentile"] = ranked["percentile"]
    out["price_method"] = current["price_method"]
    return out[columns]


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
