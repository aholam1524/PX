"""Mortgage affordability helpers (pure functions, no Streamlit)."""

from __future__ import annotations

import math

BUDGET_FIT_WITHIN = "within"
BUDGET_FIT_OVER = "over"
BUDGET_FIT_STRETCH = "stretch"

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
