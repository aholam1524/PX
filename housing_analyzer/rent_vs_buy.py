"""Rent vs buy comparison helpers (pure functions, no Streamlit)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from housing_analyzer.affordability import (
    down_payment_from_inputs,
    loan_amount,
    monthly_payment,
    typical_dwelling_price,
)
from housing_analyzer.analysis.metrics import summarize_area
from housing_analyzer.data.municipalities import (
    municipality_code_from_properties,
    municipality_metrics,
    municipality_year_for_quarter,
)
from housing_analyzer.data.rents import rent_area_for_municipality
from housing_analyzer.map import (
    METRIC_GROSS_RENTAL_YIELD,
    METRIC_PRICE,
    format_hover_text,
    format_metric_value,
    metric_is_missing,
    prepare_map_dataframe,
    resolve_building_type_label,
)
from housing_analyzer.panel import boundary_properties

FUNDING_NON_SUBSIDISED_CODE = "1"
FUNDING_NON_SUBSIDISED_LABEL = "Non-subsidised"

RENT_VS_BUY_HONESTY = (
    "Average rents are published only for regions and selected large cities, from 2025 "
    "onward. The figures below are rough regional comparisons, not statements about a "
    "single street or building."
)

RENT_VS_BUY_DISCLAIMER = (
    "Illustration only — not financial advice. Loan payments include repaying principal, "
    "which builds equity, so a higher monthly payment for owning is not simply "
    '"more expensive" than renting.'
)

_CITY_AREA_CODE_RE = re.compile(r"^\d{3}$")
_REGION_AREA_CODE_RE = re.compile(r"^MK\d{2}$")


def rent_files_available() -> bool:
    from housing_analyzer.data import paths as data_paths

    if data_paths.use_fixtures():
        return data_paths.RENTS_FIXTURE_FILE.is_file()
    return data_paths.RENTS_SNAPSHOT_FILE.is_file()


def _positive_or_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return float("nan")
    if pd.isna(value):
        return float("nan")
    number = float(value)
    if number <= 0:
        return float("nan")
    return number


def gross_rental_yield(
    rent_per_sqm_month: float | None, price_per_sqm: float | None
) -> float:
    """Annual rent divided by price, as a fraction (not a percentage)."""
    rent = _positive_or_nan(rent_per_sqm_month)
    price = _positive_or_nan(price_per_sqm)
    if math.isnan(rent) or math.isnan(price):
        return float("nan")
    return rent * 12.0 / price


def price_to_rent_ratio(
    rent_per_sqm_month: float | None, price_per_sqm: float | None
) -> float:
    rent = _positive_or_nan(rent_per_sqm_month)
    price = _positive_or_nan(price_per_sqm)
    if math.isnan(rent) or math.isnan(price):
        return float("nan")
    annual_rent = rent * 12.0
    return price / annual_rent


def rent_funding_selection() -> tuple[str, str]:
    """Non-subsidised funding class used for comparisons."""
    return (
        FUNDING_NON_SUBSIDISED_CODE,
        f"{FUNDING_NON_SUBSIDISED_CODE} — {FUNDING_NON_SUBSIDISED_LABEL}",
    )


def rent_rooms_for_building_type(
    building_type_code: str | None,
) -> tuple[str, str]:
    """Map sidebar building-type codes to rent-table room classes."""
    mapping: dict[str | None, tuple[str, str]] = {
        "1": ("1", "One-room flat"),
        "2": ("2", "Two-room flat"),
        "3": ("3", "Three-room flat+"),
        "5": ("SSS", "Total"),
        "all": ("SSS", "Total"),
        None: ("SSS", "Total"),
    }
    key = None if building_type_code in (None, "all") else building_type_code
    return mapping.get(key, ("SSS", "Total"))


def _code_from_labeled_column(value: str) -> str:
    text = str(value).strip()
    if " — " in text:
        return text.split(" — ", 1)[0].strip()
    return text


def rent_area_display_name(area_code: str, area_name: str) -> str:
    """Human label for a rent area row, including coarse geography."""
    name = (area_name or area_code).strip()
    if _REGION_AREA_CODE_RE.match(area_code):
        return f"{name} (region)"
    if area_code in {"pks", "msu"}:
        return f"{name} (aggregate)"
    if _CITY_AREA_CODE_RE.match(area_code):
        return f"{name} (city)"
    return name


def latest_rent_quarter(price_quarter: str, rents_df: pd.DataFrame) -> str | None:
    """Pick a rent quarter at or before the price quarter, else the latest available."""
    if rents_df.empty or "quarter" not in rents_df.columns:
        return None
    quarters = sorted(rents_df["quarter"].astype(str).unique())
    at_or_before = [q for q in quarters if q <= price_quarter]
    if at_or_before:
        return at_or_before[-1]
    return quarters[-1] if quarters else None


@dataclass(frozen=True)
class RentQuote:
    area_code: str
    area_name: str
    area_label: str
    quarter: str
    rent_per_sqm: float
    rent_observations: int | None
    funding_display: str
    rooms_display: str


def lookup_rent_quote(
    rents_df: pd.DataFrame,
    *,
    rent_area_code: str,
    quarter: str,
    rooms_code: str,
    funding_code: str = FUNDING_NON_SUBSIDISED_CODE,
) -> RentQuote | None:
    if rents_df.empty:
        return None
    funding_display, _ = rent_funding_selection()
    rooms_prefix = f"{rooms_code} —"
    subset = rents_df.loc[
        (rents_df["area_code"].astype(str) == str(rent_area_code))
        & (rents_df["quarter"].astype(str) == str(quarter))
        & (rents_df["funding"].astype(str).str.startswith(f"{funding_code} —"))
        & (
            rents_df["rooms"].astype(str).str.startswith(rooms_prefix)
            if rooms_code != "SSS"
            else rents_df["rooms"].astype(str).str.startswith("SSS")
        )
    ]
    if subset.empty:
        return None
    row = subset.iloc[0]
    rent = _positive_or_nan(row.get("rent_per_sqm"))
    if math.isnan(rent):
        return None
    obs = row.get("rent_observations")
    obs_int: int | None
    if obs is None or pd.isna(obs):
        obs_int = None
    else:
        obs_int = int(obs)
    area_name = str(row.get("area_name") or rent_area_code)
    return RentQuote(
        area_code=str(rent_area_code),
        area_name=area_name,
        area_label=rent_area_display_name(str(rent_area_code), area_name),
        quarter=str(quarter),
        rent_per_sqm=rent,
        rent_observations=obs_int,
        funding_display=funding_display,
        rooms_display=str(row.get("rooms") or rooms_code),
    )


@dataclass(frozen=True)
class PriceQuote:
    price_per_sqm: float
    source: str  # "postal" | "municipality"
    source_label: str


def resolve_price_per_sqm(
    prices_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    postal_code: str,
    quarter: str,
    building_type_code: str | None,
    *,
    municipality_prices: pd.DataFrame | None = None,
) -> PriceQuote | None:
    code = str(postal_code).zfill(5)
    bt_label = resolve_building_type_label(prices_df, building_type_code)
    summary = summarize_area(prices_df, code, quarter, building_type=bt_label)
    price = summary.get("price_per_sqm")
    if price is not None and not (isinstance(price, float) and math.isnan(price)) and not pd.isna(
        price
    ):
        return PriceQuote(
            price_per_sqm=float(price),
            source="postal",
            source_label="Postal-code average",
        )

    if municipality_prices is None or municipality_prices.empty:
        return None

    props = boundary_properties(boundaries, code)
    municipality_code = municipality_code_from_properties(props)
    if municipality_code is None:
        return None

    year_choice = municipality_year_for_quarter(municipality_prices, quarter)
    mun_code, _approx = _municipality_building_type(building_type_code)
    metrics = municipality_metrics(
        municipality_prices, year_choice.year, building_type=mun_code
    )
    if metrics.empty:
        return None
    match = metrics.loc[metrics["municipality_code"].astype(str) == municipality_code]
    if match.empty:
        return None
    mun_price = match.iloc[0]["price_per_sqm"]
    if mun_price is None or pd.isna(mun_price):
        return None
    mun_name = str(match.iloc[0].get("municipality_name") or municipality_code)
    return PriceQuote(
        price_per_sqm=float(mun_price),
        source="municipality",
        source_label=f"Municipality average ({mun_name}, {year_choice.year})",
    )


def _municipality_building_type(building_type_code: str | None) -> tuple[str | None, bool]:
    from housing_analyzer.hybrid_map import municipality_building_type_for_map

    return municipality_building_type_for_map(building_type_code)


@dataclass(frozen=True)
class MonthlyComparison:
    monthly_rent: float
    monthly_loan_payment: float
    monthly_charges: float
    total_monthly_owning: float
    difference_rent_minus_owning: float


def monthly_comparison(
    size_sqm: float,
    price_per_sqm: float,
    rent_per_sqm_month: float,
    down_payment: float,
    annual_rate_pct: float,
    years: float,
    monthly_charge_per_sqm: float,
) -> MonthlyComparison | None:
    price = _positive_or_nan(price_per_sqm)
    rent = _positive_or_nan(rent_per_sqm_month)
    if math.isnan(price) or math.isnan(rent) or size_sqm <= 0:
        return None
    if monthly_charge_per_sqm < 0:
        return None

    purchase = typical_dwelling_price(price, size_sqm)
    if down_payment > purchase:
        return None
    principal = loan_amount(purchase, down_payment)
    payment = monthly_payment(principal, annual_rate_pct, years)
    monthly_rent = rent * size_sqm
    monthly_charges = monthly_charge_per_sqm * size_sqm
    total_owning = payment + monthly_charges
    return MonthlyComparison(
        monthly_rent=monthly_rent,
        monthly_loan_payment=payment,
        monthly_charges=monthly_charges,
        total_monthly_owning=total_owning,
        difference_rent_minus_owning=monthly_rent - total_owning,
    )


def monthly_comparison_table_rows(
    comparison: MonthlyComparison,
) -> list[dict[str, Any]]:
    return [
        {"Item": "Monthly rent", "EUR/month": comparison.monthly_rent},
        {"Item": "Monthly loan payment (principal + interest)", "EUR/month": comparison.monthly_loan_payment},
        {"Item": "Monthly housing-company charges (assumption)", "EUR/month": comparison.monthly_charges},
        {"Item": "Total monthly cost of owning", "EUR/month": comparison.total_monthly_owning},
        {
            "Item": "Rent minus owning (negative means owning costs less cash)",
            "EUR/month": comparison.difference_rent_minus_owning,
        },
    ]


def rent_vs_buy_conclusion(
    *,
    yield_fraction: float,
    price_rent_ratio: float,
    comparison: MonthlyComparison | None,
) -> str:
    parts: list[str] = []
    if not math.isnan(yield_fraction):
        parts.append(
            f"Gross rental yield is about **{yield_fraction * 100:.1f} %** of the price per year."
        )
    if not math.isnan(price_rent_ratio):
        parts.append(
            f"The price is about **{price_rent_ratio:.1f} years** of annual rent at these averages."
        )
    if comparison is not None:
        diff = comparison.difference_rent_minus_owning
        if diff > 50:
            parts.append(
                "On these assumptions, **renting costs more per month** than owning "
                "(loan payment plus charges), but part of the loan payment builds equity."
            )
        elif diff < -50:
            parts.append(
                "On these assumptions, **owning costs more per month** than renting, "
                "though loan principal repayment is not lost spending."
            )
        else:
            parts.append(
                "Monthly rent and the estimated cost of owning are **similar** on these assumptions."
            )
    if not parts:
        return "Not enough price and rent data to compare renting and buying for this area."
    return " ".join(parts)


def _rent_for_postal_code(
    postal_code: str,
    boundaries: Mapping[str, Any],
    rents_df: pd.DataFrame,
    region_map: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
) -> RentQuote | None:
    props = boundary_properties(boundaries, postal_code)
    municipality_code = municipality_code_from_properties(props)
    if municipality_code is None:
        return None
    rent_area = rent_area_for_municipality(municipality_code, region_map=region_map)
    rent_quarter = latest_rent_quarter(quarter, rents_df)
    if rent_quarter is None:
        return None
    rooms_code, _rooms_label = rent_rooms_for_building_type(building_type_code)
    return lookup_rent_quote(
        rents_df,
        rent_area_code=rent_area,
        quarter=rent_quarter,
        rooms_code=rooms_code,
    )


def prepare_gross_rental_yield_dataframe(
    prices_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    quarter: str,
    building_type_code: str | None,
    rents_df: pd.DataFrame,
    region_map: pd.DataFrame,
    *,
    cpi_df: pd.DataFrame | None = None,
    demographics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Postal-code map layer: gross rental yield using region/city level rents."""
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

    rent_quarter = latest_rent_quarter(quarter, rents_df)
    rooms_code, _ = rent_rooms_for_building_type(building_type_code)
    yields: list[float] = []
    hovers: list[str] = []

    for record in base.to_dict("records"):
        postal = str(record["postal_code"]).zfill(5)
        price = record.get("price_per_sqm")
        quote = None
        if rent_quarter is not None:
            props = boundary_properties(boundaries, postal)
            municipality_code = municipality_code_from_properties(props)
            if municipality_code is not None:
                rent_area = rent_area_for_municipality(
                    municipality_code, region_map=region_map
                )
                quote = lookup_rent_quote(
                    rents_df,
                    rent_area_code=rent_area,
                    quarter=rent_quarter,
                    rooms_code=rooms_code,
                )
        yield_frac = gross_rental_yield(
            quote.rent_per_sqm if quote else None,
            price if price is not None else None,
        )
        yield_pct = yield_frac * 100.0 if not math.isnan(yield_frac) else float("nan")
        yields.append(yield_pct)

        if math.isnan(yield_pct):
            hovers.append(format_hover_text(record, METRIC_PRICE))
        else:
            name = record.get("area_name") or ""
            rent_note = (
                f"Rent area: {quote.area_label} ({quote.quarter})"
                if quote
                else "Rent: missing"
            )
            hovers.append(
                "<br>".join(
                    [
                        f"<b>{postal}</b> {name}",
                        f"Gross rental yield: {format_metric_value(yield_pct, METRIC_GROSS_RENTAL_YIELD)}",
                        rent_note,
                        f"Price per m²: {format_metric_value(float(price), METRIC_PRICE)}"
                        if price is not None and not pd.isna(price)
                        else "Price per m²: —",
                    ]
                )
            )

    out = base.copy()
    out[METRIC_GROSS_RENTAL_YIELD] = yields
    out["hover"] = hovers
    out["missing"] = [
        metric_is_missing({"gross_rental_yield_pct": y}, METRIC_GROSS_RENTAL_YIELD)
        or metric_is_missing(row, METRIC_PRICE)
        for y, row in zip(yields, base.to_dict("records"), strict=True)
    ]
    return out


def build_rent_vs_buy_summary(
    prices_df: pd.DataFrame,
    boundaries: Mapping[str, Any],
    postal_code: str,
    quarter: str,
    building_type_code: str | None,
    rents_df: pd.DataFrame,
    region_map: pd.DataFrame,
    *,
    size_sqm: float,
    down_payment_value: float,
    use_percent_down: bool,
    annual_rate_pct: float,
    years: float,
    monthly_charge_per_sqm: float,
    municipality_prices: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Bundle tab fields for one postal-code area."""
    code = str(postal_code).zfill(5)
    price_quote = resolve_price_per_sqm(
        prices_df,
        boundaries,
        code,
        quarter,
        building_type_code,
        municipality_prices=municipality_prices,
    )
    rent_quote = _rent_for_postal_code(
        code, boundaries, rents_df, region_map, quarter, building_type_code
    )
    funding_code, funding_display = rent_funding_selection()
    rooms_code, rooms_label = rent_rooms_for_building_type(building_type_code)

    price_per_sqm = price_quote.price_per_sqm if price_quote else float("nan")
    rent_per_sqm = rent_quote.rent_per_sqm if rent_quote else float("nan")
    yield_frac = gross_rental_yield(rent_per_sqm, price_per_sqm)
    ptr = price_to_rent_ratio(rent_per_sqm, price_per_sqm)

    comparison: MonthlyComparison | None = None
    if price_quote and rent_quote:
        typical = typical_dwelling_price(price_quote.price_per_sqm, size_sqm)
        down = down_payment_from_inputs(
            typical, down_payment_value, use_percent=use_percent_down
        )
        comparison = monthly_comparison(
            size_sqm,
            price_quote.price_per_sqm,
            rent_quote.rent_per_sqm,
            down,
            annual_rate_pct,
            years,
            monthly_charge_per_sqm,
        )

    return {
        "postal_code": code,
        "price_quote": price_quote,
        "rent_quote": rent_quote,
        "yield_fraction": yield_frac,
        "price_to_rent_ratio": ptr,
        "comparison": comparison,
        "funding_display": funding_display,
        "rooms_label": rooms_label,
        "rooms_code": rooms_code,
        "conclusion": rent_vs_buy_conclusion(
            yield_fraction=yield_frac,
            price_rent_ratio=ptr,
            comparison=comparison,
        ),
    }
