"""Mortgage affordability helpers (pure functions, no Streamlit)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from housing_analyzer.analysis.metrics import summarize_areas

BUDGET_FIT_WITHIN = "within"
BUDGET_FIT_OVER = "over"
BUDGET_FIT_STRETCH = "stretch"

# Rules of thumb for payment share of household income (not official lending limits).
PAYMENT_SHARE_COMFORTABLE = "comfortable"
PAYMENT_SHARE_STRETCHED = "stretched"
PAYMENT_SHARE_HARD = "hard"
PAYMENT_SHARE_NO_DATA = "no data"
PAYMENT_SHARE_COMFORTABLE_MAX = 0.30
PAYMENT_SHARE_STRETCHED_MAX = 0.40

PAYMENT_SHARE_LABELS: dict[str, str] = {
    PAYMENT_SHARE_COMFORTABLE: "Comfortable (≤30 % of income)",
    PAYMENT_SHARE_STRETCHED: "Stretched (30–40 %)",
    PAYMENT_SHARE_HARD: "Hard (>40 %)",
    PAYMENT_SHARE_NO_DATA: "No income data",
}

PAYMENT_SHARE_RULES_NOTE = (
    "Payment share classes (≤30 % comfortable, 30–40 % stretched, above 40 % hard) "
    "are **rules of thumb**, not official lending limits; lenders apply their own criteria."
)

_STRETCH_UPPER_RATIO = 1.2


def _months(years: float) -> int:
    if years <= 0:
        raise ValueError("years must be positive")
    months = round(years * 12)
    if months <= 0:
        raise ValueError("loan term must be at least one month")
    return months


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must not be negative")


def loan_amount(price: float, down_payment: float) -> float:
    """Principal borrowed after the down payment."""
    _validate_non_negative("price", price)
    _validate_non_negative("down_payment", down_payment)
    if down_payment > price:
        raise ValueError("down payment must not exceed price")
    return price - down_payment


def monthly_payment(principal: float, annual_rate_pct: float, years: float) -> float:
    """Standard annuity payment; at 0% rate, principal divided by term in months."""
    _validate_non_negative("principal", principal)
    _validate_non_negative("annual_rate_pct", annual_rate_pct)
    months = _months(years)
    if principal == 0.0:
        return 0.0
    if annual_rate_pct == 0.0:
        return principal / months
    monthly_rate = annual_rate_pct / 100.0 / 12.0
    factor = (1.0 + monthly_rate) ** months
    return principal * monthly_rate * factor / (factor - 1.0)


def total_interest(principal: float, annual_rate_pct: float, years: float) -> float:
    """Total interest paid over the full loan term."""
    months = _months(years)
    payment = monthly_payment(principal, annual_rate_pct, years)
    return payment * months - principal


def max_loan_from_payment(
    monthly_budget: float, annual_rate_pct: float, years: float
) -> float:
    """Largest principal whose monthly payment fits ``monthly_budget``."""
    _validate_non_negative("monthly_budget", monthly_budget)
    _validate_non_negative("annual_rate_pct", annual_rate_pct)
    months = _months(years)
    if monthly_budget == 0.0:
        return 0.0
    if annual_rate_pct == 0.0:
        return monthly_budget * months
    monthly_rate = annual_rate_pct / 100.0 / 12.0
    factor = (1.0 + monthly_rate) ** months
    return monthly_budget * (factor - 1.0) / (monthly_rate * factor)


def max_price(
    monthly_budget: float,
    down_payment: float,
    annual_rate_pct: float,
    years: float,
) -> float:
    """Highest purchase price whose loan payment fits ``monthly_budget``."""
    _validate_non_negative("down_payment", down_payment)
    return down_payment + max_loan_from_payment(
        monthly_budget, annual_rate_pct, years
    )


def typical_dwelling_price(price_per_sqm: float, size_sqm: float) -> float:
    """Nominal dwelling price from area EUR/m² and floor area."""
    _validate_non_negative("price_per_sqm", price_per_sqm)
    _validate_non_negative("size_sqm", size_sqm)
    return price_per_sqm * size_sqm


def price_to_budget_ratio(typical_price: float, max_affordable_price: float) -> float:
    """Ratio of typical area price to the user's max affordable price."""
    if max_affordable_price <= 0:
        raise ValueError("max_affordable_price must be positive")
    if math.isnan(typical_price):
        return float("nan")
    return typical_price / max_affordable_price


def budget_ratio_and_category(
    typical_price: float, max_affordable_price: float
) -> tuple[float, str]:
    """Ratio and fit category for one dwelling price vs max affordable price."""
    if max_affordable_price <= 0:
        return float("inf"), BUDGET_FIT_OVER
    ratio = price_to_budget_ratio(typical_price, max_affordable_price)
    return ratio, classify_budget_fit_ratio(ratio)


def payment_to_income(
    monthly_payment: float, annual_household_income: float
) -> float:
    """Monthly payment × 12 / annual household income, as a fraction."""
    if (
        annual_household_income is None
        or (isinstance(annual_household_income, float) and np.isnan(annual_household_income))
        or pd.isna(annual_household_income)
        or annual_household_income <= 0
    ):
        return float("nan")
    if (
        monthly_payment is None
        or (isinstance(monthly_payment, float) and np.isnan(monthly_payment))
        or pd.isna(monthly_payment)
    ):
        return float("nan")
    return float(monthly_payment) * 12.0 / float(annual_household_income)


def classify_payment_share(share: float) -> str:
    """Rules of thumb: ≤30 % comfortable, ≤40 % stretched, above hard; missing → no data."""
    if share is None or (isinstance(share, float) and (np.isnan(share) or math.isinf(share))):
        return PAYMENT_SHARE_NO_DATA
    if pd.isna(share):
        return PAYMENT_SHARE_NO_DATA
    value = float(share)
    if value <= PAYMENT_SHARE_COMFORTABLE_MAX:
        return PAYMENT_SHARE_COMFORTABLE
    if value <= PAYMENT_SHARE_STRETCHED_MAX:
        return PAYMENT_SHARE_STRETCHED
    return PAYMENT_SHARE_HARD


def stress_test(
    price: float,
    down_payment: float,
    annual_rate_pct: float,
    years: float,
    annual_household_income: float,
    *,
    rate_steps: tuple[int, ...] = (0, 1, 2, 3),
    income_shock_pct: float = 0,
) -> pd.DataFrame:
    """One row per rate step (percentage points added) with payment share and class."""
    shocked_income = annual_household_income
    if (
        shocked_income is not None
        and not (isinstance(shocked_income, float) and np.isnan(shocked_income))
        and not pd.isna(shocked_income)
        and income_shock_pct != 0
    ):
        shocked_income = float(shocked_income) * (1.0 + float(income_shock_pct) / 100.0)

    rows: list[dict[str, Any]] = []
    for step in rate_steps:
        rate = float(annual_rate_pct) + float(step)
        try:
            principal = loan_amount(price, down_payment)
            payment = monthly_payment(principal, rate, years)
        except ValueError:
            payment = float("nan")
        share = payment_to_income(payment, shocked_income)
        rows.append(
            {
                "rate_step_pp": int(step),
                "annual_rate_pct": rate,
                "monthly_payment": payment,
                "payment_income_share": share,
                "payment_share_class": classify_payment_share(share),
            }
        )
    return pd.DataFrame(rows)


def classify_budget_fit_ratio(ratio: float) -> str:
    """Map price/budget ratio to fit, stretch (≤20% over), or over."""
    if ratio != ratio or math.isinf(ratio):  # NaN
        return BUDGET_FIT_OVER
    if ratio <= 1.0:
        return BUDGET_FIT_WITHIN
    if ratio <= _STRETCH_UPPER_RATIO:
        return BUDGET_FIT_STRETCH
    return BUDGET_FIT_OVER


def down_payment_from_inputs(
    price: float,
    down_payment_value: float,
    *,
    use_percent: bool,
) -> float:
    """Euro down payment from a euro amount or a percentage of ``price``."""
    _validate_non_negative("price", price)
    _validate_non_negative("down_payment_value", down_payment_value)
    if use_percent:
        if down_payment_value > 100.0:
            raise ValueError("down payment percent must not exceed 100")
        return price * down_payment_value / 100.0
    return down_payment_value


AFFORDABILITY_DISCLAIMER = (
    "This calculator ignores maintenance charges, taxes, and future interest-rate "
    "changes. The figures are an illustration only — not financial advice or a loan offer."
)

_BUDGET_FIT_SORT_ORDER = {
    BUDGET_FIT_WITHIN: 0,
    BUDGET_FIT_STRETCH: 1,
    BUDGET_FIT_OVER: 2,
}


def _household_income_lookup(
    demographics_df: pd.DataFrame | None,
) -> tuple[dict[str, float], dict[str, int | float]]:
    """Postal code → median household income (EUR/year) and data year."""
    if demographics_df is None or demographics_df.empty:
        return {}, {}
    demo = demographics_df.copy()
    demo["postal_code"] = demo["postal_code"].astype(str).str.zfill(5)
    income_by_code: dict[str, float] = {}
    year_by_code: dict[str, int | float] = {}
    for _, row in demo.iterrows():
        code = str(row["postal_code"]).zfill(5)
        raw = row.get("median_household_income_eur")
        if raw is None or (isinstance(raw, float) and np.isnan(raw)) or pd.isna(raw):
            continue
        income_by_code[code] = float(raw)
        year = row.get("data_year")
        if year is not None and not pd.isna(year):
            year_by_code[code] = year
    return income_by_code, year_by_code


def affordability_table(
    prices_df: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
    size_sqm: float,
    max_affordable_price: float,
    annual_rate_pct: float,
    years: float,
    down_payment_value: float,
    use_percent: bool,
    demographics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per postal-code area with a published price for the selection."""
    from housing_analyzer.map import resolve_building_type_label
    from housing_analyzer.panel import postal_code_name_lookup

    bt_label = resolve_building_type_label(prices_df, building_type_code)
    summaries = summarize_areas(prices_df, quarter, building_type=bt_label)
    if summaries.empty:
        return pd.DataFrame(
            columns=[
                "postal_code",
                "area_name",
                "municipality",
                "typical_price",
                "price_per_sqm",
                "budget_fit",
                "budget_ratio",
                "monthly_payment",
                "headroom",
                "pct_change_1y",
                "reliability",
                "median_household_income_eur",
                "household_income_data_year",
                "payment_income_share",
                "payment_share_class",
            ]
        )

    income_by_code, year_by_code = _household_income_lookup(demographics_df)
    name_lookup = postal_code_name_lookup(prices_df)
    rows: list[dict[str, Any]] = []
    for postal_code, summary in summaries.iterrows():
        price_sqm = summary["price_per_sqm"]
        if price_sqm is None or (isinstance(price_sqm, float) and np.isnan(price_sqm)):
            continue
        price_sqm_f = float(price_sqm)
        typical = typical_dwelling_price(price_sqm_f, size_sqm)
        ratio, category = budget_ratio_and_category(typical, max_affordable_price)
        down = down_payment_from_inputs(
            typical, down_payment_value, use_percent=use_percent
        )
        if down > typical:
            # A fixed-EUR down payment can exceed a cheap area's typical price;
            # loan_amount() would raise, so treat it as over budget instead.
            category = BUDGET_FIT_OVER
            payment = float("nan")
        else:
            principal = loan_amount(typical, down)
            payment = monthly_payment(principal, annual_rate_pct, years)
        headroom = max_affordable_price - typical
        code = str(postal_code).zfill(5)
        area_name, municipality = name_lookup.get(code, ("", ""))
        hh_income = income_by_code.get(code)
        share = payment_to_income(payment, hh_income) if hh_income is not None else float("nan")
        rows.append(
            {
                "postal_code": code,
                "area_name": area_name,
                "municipality": municipality,
                "typical_price": typical,
                "price_per_sqm": price_sqm_f,
                "budget_fit": category,
                "budget_ratio": ratio,
                "monthly_payment": payment,
                "headroom": headroom,
                "pct_change_1y": summary.get("pct_change_1y"),
                "reliability": summary.get("reliability"),
                "median_household_income_eur": hh_income,
                "household_income_data_year": year_by_code.get(code),
                "payment_income_share": share,
                "payment_share_class": classify_payment_share(share),
            }
        )
    return pd.DataFrame(rows)


def affordability_summary(
    table: pd.DataFrame, *, max_affordable_price: float
) -> dict[str, float | int]:
    """Counts and shares for areas that fit within the max affordable price."""
    n_with_price = len(table)
    n_fits = int((table["budget_fit"] == BUDGET_FIT_WITHIN).sum()) if n_with_price else 0
    share = (100.0 * n_fits / n_with_price) if n_with_price else 0.0
    n_comfortable = (
        int((table["payment_share_class"] == PAYMENT_SHARE_COMFORTABLE).sum())
        if n_with_price and "payment_share_class" in table.columns
        else 0
    )
    return {
        "n_with_price": n_with_price,
        "n_fits": n_fits,
        "fit_share_pct": share,
        "n_comfortable": n_comfortable,
        "max_affordable_price": max_affordable_price,
    }


def filter_affordability_table(
    table: pd.DataFrame,
    *,
    only_fits: bool = False,
    only_comfortable: bool = False,
    hide_low_reliability: bool = False,
    municipalities: list[str] | None = None,
) -> pd.DataFrame:
    """Apply UI filters without mutating the input frame."""
    out = table
    if only_fits:
        out = out.loc[out["budget_fit"] == BUDGET_FIT_WITHIN]
    if only_comfortable:
        out = out.loc[out["payment_share_class"] == PAYMENT_SHARE_COMFORTABLE]
    if hide_low_reliability:
        out = out.loc[out["reliability"] != "low"]
    if municipalities:
        allowed = {m.strip() for m in municipalities if m and str(m).strip()}
        if allowed:
            out = out.loc[out["municipality"].isin(allowed)]
    return out.copy()


def sort_affordability_table(table: pd.DataFrame) -> pd.DataFrame:
    """Fitting areas first, then ascending price per m²."""
    if table.empty:
        return table.copy()
    order = table["budget_fit"].map(_BUDGET_FIT_SORT_ORDER).fillna(99)
    return (
        table.assign(_fit_order=order)
        .sort_values(["_fit_order", "price_per_sqm", "postal_code"], kind="mergesort")
        .drop(columns="_fit_order")
        .reset_index(drop=True)
    )
