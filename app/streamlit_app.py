"""Streamlit UI for the housing price analyzer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import streamlit as st

from housing_analyzer.analysis.compare import (
    area_catalog,
    build_compare_chart_data,
    build_compare_figure,
    build_comparison_table,
    search_area_catalog,
    summaries_for_compare,
)
from housing_analyzer.analysis.metrics import summarize_area
from housing_analyzer.analysis.relationships import (
    CORRELATION_DISCLAIMER,
    build_relationships_summary,
)
from housing_analyzer.analysis.similar_areas import similar_areas, similar_areas_explanation
from housing_analyzer.data import (
    load_boundaries,
    load_cpi,
    load_demographics,
    load_manifest,
    load_prices,
)
from housing_analyzer.data.demographics import (
    attach_demographics_to_summaries,
    load_national_demographics,
)
from housing_analyzer.data.cpi import cpi_by_quarter
from housing_analyzer.data.paths import use_fixtures
from housing_analyzer.data.snapshot import snapshot_is_complete
from housing_analyzer.home_value import (
    HOME_VALUE_DISCLAIMER,
    estimate_home_value,
    indexed_area_chart_from_purchase,
)
from housing_analyzer.affordability import (
    AFFORDABILITY_DISCLAIMER,
    affordability_summary,
    affordability_table,
    down_payment_from_inputs,
    filter_affordability_table,
    loan_amount,
    max_price,
    monthly_payment,
    sort_affordability_table,
    total_interest,
)
from housing_analyzer.hybrid_map import (
    building_type_mapping_caption,
    build_hybrid_choropleth_figure,
    classify_hybrid_postal_coverage,
    hybrid_coverage_counts,
    hybrid_metric_color_range,
    hybrid_metric_supported,
    hybrid_unsupported_reason,
    map_selection_from_event_or_postal,
    municipality_files_available,
    municipality_map_card_data,
    prepare_municipality_map_dataframe,
    postal_to_municipality_codes,
)
from housing_analyzer.data.municipalities import load_municipality_boundaries, load_municipality_prices
from housing_analyzer.map import (
    BUDGET_FIT_LABELS,
    BUILDING_TYPE_CHOICES,
    METRIC_CHOICES,
    METRIC_CHANGE_1Y,
    METRIC_CHANGE_5Y,
    METRIC_FITS_BUDGET,
    METRIC_MARKET_ACTIVITY,
    METRIC_PRICE,
    build_choropleth_figure,
    count_areas_with_published_price,
    default_map_quarter,
    format_metric_value,
    list_quarters,
    metric_color_range,
    postal_code_from_selection,
    prepare_budget_fit_dataframe,
    plotly_map_chart_config,
    prepare_map_dataframe,
    quarter_meets_coverage_threshold,
    reliability_display,
    resolve_building_type_label,
    search_area_matches,
    trailing_sales_by_area,
    typical_quarter_price_coverage,
)
from housing_analyzer.panel import (
    area_detail_export_frame,
    area_display_name,
    area_profile_data_year,
    build_area_profile_items,
    build_sales_volume_figure,
    build_trend_chart_data,
    build_trend_figure,
    building_type_panel_options,
    default_selected_postal_code,
    flag_unusual_quarter_changes,
    format_area_header,
    market_activity_for_area,
    market_activity_panel_caption,
    MARKET_ACTIVITY_METRIC_HELP,
    municipality_name_from_prices,
    quarterly_area_prices,
    quarterly_transaction_counts,
    reliability_explanation,
    resolve_panel_building_type,
    trailing_sales_count,
)

MAX_COMPARE_AREAS = 4

st.set_page_config(page_title="Housing price analyzer", layout="wide")

st.title("Housing price analyzer")
st.write(
    "Explore Finnish postal-code areas on a map coloured by housing price metrics, "
    "with trends and comparisons for a selected area, and cross-area relationship charts."
)
st.caption("For information only — not investment advice.")

if not use_fixtures() and not snapshot_is_complete():
    st.error(
        "The housing data snapshot is not in this checkout yet, so the app cannot load "
        "prices or map boundaries without a long live download."
    )
    st.info(
        "Repository owners: run the **Refresh housing data** workflow in GitHub Actions "
        "(Actions → Refresh housing data → Run workflow). That job rebuilds "
        "`data/snapshot/` and opens a pull request; merge it so hosted and cold starts "
        "use the committed snapshot."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def _load_housing_data() -> tuple:
    prices = load_prices()
    boundaries = load_boundaries()
    cpi = load_cpi()
    demographics = load_demographics()
    manifest = load_manifest()
    return prices, boundaries, cpi, demographics, manifest


@st.cache_data(show_spinner=False)
def _load_national_demographics() -> pd.Series:
    return load_national_demographics()


@st.cache_data(show_spinner=False)
def _load_municipality_data() -> tuple[pd.DataFrame | None, dict | None]:
    if not municipality_files_available():
        return None, None
    try:
        return load_municipality_prices(), load_municipality_boundaries()
    except Exception:  # noqa: BLE001 — fall back to postal-only map
        return None, None


@st.cache_data(show_spinner=False)
def _cached_map_frame(
    _prices,
    _boundaries,
    _cpi,
    _demographics,
    quarter,
    building_type_code,
    metric,
    size_sqm,
    max_affordable_price,
    annual_rate_pct,
    loan_years,
    down_payment_value,
    down_use_percent,
):
    """Map table for one selection. The large inputs are not hashed (leading underscore);
    the selection values are the cache key, so clicking the map does not recompute it."""
    if metric == METRIC_FITS_BUDGET:
        return prepare_budget_fit_dataframe(
            _prices,
            _boundaries,
            quarter,
            building_type_code,
            size_sqm,
            max_affordable_price,
            annual_rate_pct=annual_rate_pct,
            years=loan_years,
            down_payment_value=down_payment_value,
            use_percent=down_use_percent,
            cpi_df=_cpi,
            demographics_df=_demographics,
        )
    return prepare_map_dataframe(
        _prices,
        _boundaries,
        quarter,
        building_type_code,
        metric,
        cpi_df=_cpi,
        demographics_df=_demographics,
    )


@st.cache_data(show_spinner=False)
def _cached_trend_chart_data(
    _prices, _boundaries, _cpi, postal_code, building_type, index_to_100, use_real
):
    """Trend chart series for one area/building-type/index selection (see _cached_map_frame)."""
    return build_trend_chart_data(
        _prices,
        _boundaries,
        postal_code,
        building_type,
        index_to_100=index_to_100,
        use_real=use_real,
        cpi_df=_cpi,
    )


@st.cache_data(show_spinner=False)
def _cached_area_prices(_prices, postal_code, building_type):
    return quarterly_area_prices(_prices, postal_code, building_type)


@st.cache_data(show_spinner=False)
def _cached_transaction_counts(_prices, postal_code, building_type):
    return quarterly_transaction_counts(_prices, postal_code, building_type)


def _friendly_load_error(exc: BaseException) -> str:
    return f"Could not load housing data: {exc}"


def _format_building_type_display(
    prices: pd.DataFrame, building_type_code: str | None
) -> str:
    if building_type_code in (None, "all"):
        return "All building types"
    label = resolve_building_type_label(prices, building_type_code)
    return label or dict(BUILDING_TYPE_CHOICES).get(building_type_code, building_type_code)


def _init_compare_session_state() -> None:
    if "compare_postal_codes" not in st.session_state:
        st.session_state.compare_postal_codes = []


def _add_to_compare(postal_code: str) -> bool:
    code = str(postal_code).zfill(5)
    current: list[str] = list(st.session_state.compare_postal_codes)
    if code in current:
        return False
    if len(current) >= MAX_COMPARE_AREAS:
        st.warning(f"Comparison is limited to {MAX_COMPARE_AREAS} areas. Remove one to add another.")
        return False
    current.append(code)
    st.session_state.compare_postal_codes = current
    # The Compare tab's multiselect owns its own widget state once created, so its
    # `default=` is ignored on reruns; update it directly or this add is lost the
    # moment `_render_compare_tab` re-renders the multiselect later in this run.
    if "compare_multiselect" in st.session_state:
        st.session_state.compare_multiselect = current
    return True


def _format_compare_option(catalog: pd.DataFrame, code: str) -> str:
    row = catalog.loc[catalog["postal_code"] == code]
    name = str(row["area_name"].iloc[0]) if not row.empty else ""
    return f"{code} — {name}".strip(" —")


def _affordability_sidebar_inputs() -> dict[str, float | bool | str]:
    st.header("Affordability")
    size_sqm = st.number_input(
        "Apartment size (m²)",
        min_value=1.0,
        value=60.0,
        step=1.0,
        key="afford_size_sqm",
    )
    monthly_budget = st.number_input(
        "Monthly budget (EUR)",
        min_value=0.0,
        value=1200.0,
        step=50.0,
        key="afford_monthly_budget",
    )
    annual_rate_pct = st.number_input(
        "Interest rate (% per year)",
        min_value=0.0,
        value=4.0,
        step=0.1,
        key="afford_rate",
    )
    loan_years = st.number_input(
        "Loan term (years)",
        min_value=1.0,
        value=25.0,
        step=1.0,
        key="afford_years",
    )
    down_mode = st.radio(
        "Down payment as",
        options=("euros", "percent"),
        format_func=lambda v: "Euros" if v == "euros" else "Percent of price",
        horizontal=True,
        key="afford_down_mode",
    )
    down_payment_value = st.number_input(
        "Down payment",
        min_value=0.0,
        value=20.0 if down_mode == "percent" else 30_000.0,
        step=500.0 if down_mode == "euros" else 1.0,
        key="afford_down_value",
    )
    return {
        "size_sqm": float(size_sqm),
        "monthly_budget": float(monthly_budget),
        "annual_rate_pct": float(annual_rate_pct),
        "loan_years": float(loan_years),
        "down_mode": down_mode,
        "down_payment_value": float(down_payment_value),
    }


def _compute_max_affordable(afford: dict[str, float | bool | str]) -> float:
    loan_cap = max_price(
        float(afford["monthly_budget"]),
        0.0,
        float(afford["annual_rate_pct"]),
        float(afford["loan_years"]),
    )
    if afford["down_mode"] == "percent":
        pct = float(afford["down_payment_value"]) / 100.0
        if pct >= 1.0:
            return 0.0
        return loan_cap / (1.0 - pct)
    return max_price(
        float(afford["monthly_budget"]),
        float(afford["down_payment_value"]),
        float(afford["annual_rate_pct"]),
        float(afford["loan_years"]),
    )


def _affordability_sidebar_caption(afford: dict[str, float | bool | str]) -> str:
    if afford["down_mode"] == "percent":
        down_part = f"{afford['down_payment_value']:g} % down payment"
    else:
        down_part = f"{afford['down_payment_value']:,.0f} EUR down payment"
    return (
        f"{afford['size_sqm']:g} m², {afford['monthly_budget']:,.0f} EUR/month, "
        f"{afford['annual_rate_pct']:g} %, {afford['loan_years']:g} years, {down_part}"
    )


@st.cache_data(show_spinner=False)
def _cached_affordability_table(
    _prices,
    quarter,
    building_type_code,
    size_sqm,
    max_affordable_price,
    annual_rate_pct,
    loan_years,
    down_payment_value,
    down_use_percent,
):
    return affordability_table(
        _prices,
        quarter,
        building_type_code,
        size_sqm,
        max_affordable_price,
        annual_rate_pct,
        loan_years,
        down_payment_value,
        down_use_percent,
    )


def _affordability_table_display_frame(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["fit"] = display["budget_fit"].map(BUDGET_FIT_LABELS)
    display["reliability"] = display["reliability"].map(reliability_display)
    return display[
        [
            "postal_code",
            "area_name",
            "municipality",
            "typical_price",
            "price_per_sqm",
            "fit",
            "monthly_payment",
            "headroom",
            "pct_change_1y",
            "reliability",
        ]
    ].rename(
        columns={
            "postal_code": "Postal code",
            "area_name": "Area name",
            "municipality": "Municipality",
            "typical_price": "Typical price (EUR)",
            "price_per_sqm": "Price per m² (EUR)",
            "fit": "Fit",
            "monthly_payment": "Monthly payment (EUR)",
            "headroom": "Headroom (EUR)",
            "pct_change_1y": "1-year change (%)",
            "reliability": "Reliability",
        }
    )


def _postal_from_affordability_selection(
    sorted_table: pd.DataFrame, selection: Any
) -> str | None:
    if selection is None:
        return None
    rows = getattr(getattr(selection, "selection", None), "rows", None)
    if not rows:
        return None
    idx = int(rows[0])
    if idx < 0 or idx >= len(sorted_table):
        return None
    return str(sorted_table.iloc[idx]["postal_code"]).zfill(5)


def _render_affordability_tab(
    prices: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
    afford: dict[str, float | bool | str],
    selected_postal_code: str | None,
) -> None:
    st.subheader("Where can I afford to live?")
    st.caption(
        f"Quarter **{quarter}** · {_format_building_type_display(prices, building_type_code)}"
    )

    use_percent = afford["down_mode"] == "percent"
    max_affordable = _compute_max_affordable(afford)
    table = _cached_affordability_table(
        prices,
        quarter,
        building_type_code,
        float(afford["size_sqm"]),
        max_affordable,
        float(afford["annual_rate_pct"]),
        float(afford["loan_years"]),
        float(afford["down_payment_value"]),
        use_percent,
    )
    summary = affordability_summary(table, max_affordable_price=max_affordable)

    st.markdown(f"**{_affordability_sidebar_caption(afford)}**; change them in the sidebar.")
    per_sqm_cap = (
        max_affordable / float(afford["size_sqm"]) if float(afford["size_sqm"]) > 0 else 0.0
    )
    if summary["n_with_price"] == 0:
        st.warning("No areas have a published price for this quarter and building type.")
    elif summary["n_fits"]:
        st.markdown(
            f"You can afford a typical flat in **{summary['n_fits']:,} of "
            f"{summary['n_with_price']:,} areas** that have a price "
            f"({summary['fit_share_pct']:.0f} %). Maximum affordable price: "
            f"**{max_affordable:,.0f} EUR** (about **{per_sqm_cap:,.0f} EUR/m²** for "
            f"{afford['size_sqm']:g} m²)."
        )
    else:
        st.markdown(
            "No areas fit your budget at typical prices for this selection. "
            "Try a **higher monthly budget**, a **smaller apartment size**, or a "
            f"**larger down payment**. Maximum affordable price: **{max_affordable:,.0f} EUR** "
            f"(about **{per_sqm_cap:,.0f} EUR/m²** for {afford['size_sqm']:g} m²)."
        )

    if (
        afford["down_mode"] == "percent"
        and float(afford["down_payment_value"]) >= 100.0
    ):
        st.caption(
            "A 100%+ down payment covers any price from savings alone, so the maximum "
            "affordable price is not limited by your monthly budget."
        )

    municipality_options = sorted(
        {m for m in table["municipality"].dropna().astype(str) if m.strip()}
    )
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        only_fits = st.checkbox("Only areas that fit", value=True, key="afford_only_fits")
    with filter_col2:
        hide_low = st.checkbox(
            "Hide low-reliability areas", value=False, key="afford_hide_low"
        )
    with filter_col3:
        municipality_filter = st.multiselect(
            "Municipality",
            options=municipality_options,
            default=[],
            key="afford_municipality_filter",
            placeholder="All municipalities",
        )

    filtered = filter_affordability_table(
        table,
        only_fits=only_fits,
        hide_low_reliability=hide_low,
        municipalities=municipality_filter or None,
    )
    sorted_table = sort_affordability_table(filtered)
    display = _affordability_table_display_frame(sorted_table)

    if sorted_table.empty:
        st.caption("No areas match the filters.")
    else:
        table_state = st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="afford_area_table",
        )
        picked = _postal_from_affordability_selection(sorted_table, table_state)
        if picked:
            st.session_state.selected_postal_code = picked
            st.session_state.selected_map_level = "postal"

    active = st.session_state.get("selected_postal_code") or selected_postal_code
    if active:
        st.caption(
            f"Selected area **{str(active).zfill(5)}** — open the **Map** tab to see it on the map."
        )

    with st.expander("How the numbers are calculated"):
        calc_source = st.radio(
            "Show loan maths for",
            options=("selected", "target"),
            format_func=lambda v: (
                "Selected area (from table or Map tab)"
                if v == "selected"
                else "A target price I enter"
            ),
            horizontal=True,
            key="afford_calc_source",
        )
        calc_price: float | None = None
        if calc_source == "target":
            calc_price = float(
                st.number_input(
                    "Target price (EUR)",
                    min_value=0.0,
                    value=float(max_affordable),
                    step=5_000.0,
                    key="afford_calc_target_price",
                )
            )
        elif active:
            code = str(active).zfill(5)
            match = table.loc[table["postal_code"] == code]
            if not match.empty:
                calc_price = float(match.iloc[0]["typical_price"])
                st.caption(
                    f"Typical price for **{code}** at {afford['size_sqm']:g} m²: "
                    f"**{calc_price:,.0f} EUR**."
                )
            else:
                st.warning(
                    f"**{code}** has no published price for this quarter and building type."
                )
        else:
            st.info("Select a row in the table above or an area on the **Map** tab.")

        if calc_price is not None:
            try:
                down = down_payment_from_inputs(
                    calc_price,
                    float(afford["down_payment_value"]),
                    use_percent=use_percent,
                )
                principal = loan_amount(calc_price, down)
                payment = monthly_payment(
                    principal,
                    float(afford["annual_rate_pct"]),
                    float(afford["loan_years"]),
                )
                interest = total_interest(
                    principal,
                    float(afford["annual_rate_pct"]),
                    float(afford["loan_years"]),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.markdown(
                    f"- Typical / target price: **{calc_price:,.0f} EUR**\n"
                    f"- Down payment: **{down:,.0f} EUR**\n"
                    f"- Loan amount: **{principal:,.0f} EUR**\n"
                    f"- Monthly payment: **{payment:,.2f} EUR**\n"
                    f"- Total interest over term: **{interest:,.0f} EUR**"
                )
        st.caption(AFFORDABILITY_DISCLAIMER)


def _render_my_home_tab(
    prices: pd.DataFrame,
    cpi: pd.DataFrame,
    selected_postal_code: str | None,
) -> None:
    st.subheader("My home")
    st.caption(
        "See how the area's average price per square metre has moved since you bought, "
        "and a rough illustration of what your home might be worth now."
    )
    st.info(HOME_VALUE_DISCLAIMER)

    catalog = area_catalog(prices)
    default_code = (
        str(selected_postal_code).zfill(5)
        if selected_postal_code
        else default_selected_postal_code(prices)
    ) or "00100"

    search_query = st.text_input(
        "Search by postal code or area name",
        placeholder="e.g. 00100 or Punavuori",
        key="my_home_search",
    )
    search_hits = search_area_catalog(catalog, search_query)
    options = sorted(set(catalog["postal_code"].astype(str).str.zfill(5)))
    if search_hits:
        options = search_hits + [c for c in options if c not in search_hits]
    default_index = options.index(default_code) if default_code in options else 0
    postal_code = st.selectbox(
        "Postal code",
        options=options,
        index=default_index,
        format_func=lambda c: _format_compare_option(catalog, c),
        key="my_home_postal",
    )

    bt_codes = [code for code, _ in building_type_panel_options()]
    bt_labels = {code: label for code, label in building_type_panel_options()}
    building_type_code = st.selectbox(
        "Building type",
        options=bt_codes,
        format_func=lambda c: bt_labels[c],
        key="my_home_building_type",
    )
    building_type = resolve_panel_building_type(prices, building_type_code)

    quarters = list_quarters(prices)
    if not quarters:
        st.warning("No quarterly price data is available.")
        return
    purchase_quarter = st.selectbox(
        "Purchase quarter",
        options=quarters,
        index=max(0, len(quarters) - 5),
        key="my_home_purchase_quarter",
    )
    purchase_price = st.number_input(
        "Purchase price (EUR)",
        min_value=0.0,
        value=250_000.0,
        step=5_000.0,
        key="my_home_purchase_price",
    )
    size_sqm = st.number_input(
        "Apartment size (m², optional — shows price per m² you paid)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="my_home_size_sqm",
    )

    cpi_quarterly = cpi_by_quarter(cpi)
    result = estimate_home_value(
        prices,
        cpi_quarterly,
        postal_code,
        building_type,
        purchase_quarter,
        float(purchase_price),
    )

    if not result.enough_data:
        st.warning(
            result.notes[0]
            if result.notes
            else "Not enough data to estimate how the area has moved since purchase."
        )
        for note in result.notes[1:]:
            st.caption(note)
        return

    if size_sqm and size_sqm > 0:
        paid_per_sqm = float(purchase_price) / float(size_sqm)
        st.caption(f"You paid about **{paid_per_sqm:,.0f} EUR/m²** ({size_sqm:g} m²).")

    st.metric("Estimated value today (illustration)", f"{result.estimated_value:,.0f} EUR")
    c1, c2 = st.columns(2)
    c1.metric("Nominal change since purchase", f"{result.nominal_change_pct:+.1f} %")
    real_label = (
        f"{result.real_change_pct:+.1f} %"
        if result.real_change_pct is not None
        else "—"
    )
    c2.metric("Real change (inflation-adjusted)", real_label)

    sales_4q = trailing_sales_count(
        prices, result.postal_code, result.latest_quarter_used or purchase_quarter, building_type_code
    )
    rel_text = reliability_explanation(result.reliability, sales_4q)
    st.caption(f"**Reliability (latest quarter):** {rel_text}")

    if result.purchase_sales_count is not None or result.latest_sales_count is not None:
        parts = []
        if result.purchase_sales_count is not None:
            parts.append(
                f"{int(result.purchase_sales_count)} sales in purchase quarter "
                f"{result.purchase_quarter_used}"
            )
        if result.latest_sales_count is not None:
            parts.append(
                f"{int(result.latest_sales_count)} sales in latest quarter "
                f"{result.latest_quarter_used}"
            )
        st.caption(" · ".join(parts))

    st.caption(
        f"Area price **{result.purchase_price_per_sqm:,.0f} EUR/m²** "
        f"({result.purchase_quarter_used}) → "
        f"**{result.latest_price_per_sqm:,.0f} EUR/m²** ({result.latest_quarter_used}); "
        f"index **{result.price_index:.3f}**."
    )

    if result.purchase_quarter_used and result.latest_quarter_used:
        fig = indexed_area_chart_from_purchase(
            prices,
            result.postal_code,
            building_type,
            result.purchase_quarter_used,
            through_quarter=result.latest_quarter_used,
        )
        st.plotly_chart(fig, use_container_width=True, key="my_home_index_chart")

    for note in result.notes:
        st.caption(note)


def _render_relationships_tab(
    prices: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
    cpi: pd.DataFrame,
    demographics: pd.DataFrame,
) -> None:
    st.subheader("Relationships")
    st.caption(
        f"Quarter **{quarter}** · {_format_building_type_display(prices, building_type_code)}"
    )
    bt_label = resolve_building_type_label(prices, building_type_code)
    cpi_quarterly = cpi_by_quarter(cpi)
    summaries = attach_demographics_to_summaries(
        summaries_for_compare(
            prices, quarter, bt_label, cpi_quarterly=cpi_quarterly
        ),
        demographics,
    )
    summary = build_relationships_summary(summaries, demographics)
    st.write(
        f"**{len(summary.included)}** areas with reliable prices and complete demographics. "
        f"**{summary.excluded_count}** areas excluded (unreliable price, missing demographics, "
        "or postal codes only in prices or Paavo)."
    )
    st.caption(CORRELATION_DISCLAIMER)
    for plot in summary.plots:
        st.plotly_chart(plot.figure, use_container_width=True, key=f"rel_{plot.spec.key}")


def _render_compare_tab(
    prices: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
    cpi: pd.DataFrame,
    demographics: pd.DataFrame,
) -> None:
    _init_compare_session_state()
    catalog = area_catalog(prices)
    bt_label = resolve_building_type_label(prices, building_type_code)

    st.subheader("Compare areas")
    st.caption(
        f"Quarter **{quarter}** · {_format_building_type_display(prices, building_type_code)}"
    )

    search_query = st.text_input(
        "Search by postal code or area name",
        placeholder="e.g. 00100 or Punavuori",
        key="compare_search",
    )
    search_hits = search_area_catalog(catalog, search_query)
    options = sorted(set(catalog["postal_code"].astype(str).str.zfill(5)))
    if search_hits:
        options = search_hits + [c for c in options if c not in search_hits]

    selected = st.multiselect(
        "Areas to compare (up to four)",
        options=options,
        default=[
            c
            for c in st.session_state.compare_postal_codes
            if c in options
        ][:MAX_COMPARE_AREAS],
        format_func=lambda c: _format_compare_option(catalog, c),
        key="compare_multiselect",
        max_selections=MAX_COMPARE_AREAS,
    )
    st.session_state.compare_postal_codes = selected[:MAX_COMPARE_AREAS]

    if not selected:
        st.info("Choose one to four areas to see the comparison table and chart.")
        return

    cpi_quarterly = cpi_by_quarter(cpi)
    summaries = attach_demographics_to_summaries(
        summaries_for_compare(
            prices, quarter, bt_label, cpi_quarterly=cpi_quarterly
        ),
        demographics,
    )
    sales = trailing_sales_by_area(prices, quarter, building_type_code)
    include_real = "pct_change_1y_real" in summaries.columns

    table = build_comparison_table(
        summaries,
        sales,
        selected,
        include_real=include_real,
    )
    st.dataframe(table, use_container_width=True)

    index_mode = st.checkbox(
        "Index to 100 at the start",
        key="compare_index",
    )
    chart_data = build_compare_chart_data(
        prices,
        selected,
        bt_label,
        index_to_100=index_mode,
        use_real=False,
        cpi_quarterly=cpi_quarterly,
    )
    st.plotly_chart(
        build_compare_figure(chart_data),
        use_container_width=True,
        key="compare_chart",
    )

    st.markdown("### Similar areas")
    st.caption(similar_areas_explanation())
    if st.session_state.get("similar_for") not in selected:
        st.session_state.similar_for = selected[0]
    similar_for = st.selectbox(
        "Similar areas for",
        options=selected,
        format_func=lambda c: _format_compare_option(catalog, c),
        key="similar_for",
    )
    matches = similar_areas(summaries, similar_for)
    if not matches:
        st.write("No similar areas with OK reliability and complete price data.")
    else:
        rows = []
        for item in matches:
            rows.append(
                {
                    "Postal code": item.postal_code,
                    "Distance": f"{item.distance:.3f}",
                    "Price per m²": f"{item.price_per_sqm:,.0f}",
                    "1y change": f"{item.pct_change_1y:+.1f}%",
                    "5y change": f"{item.pct_change_5y:+.1f}%",
                    "Rank": int(item.rank) if pd.notna(item.rank) else "—",
                    "Percentile": f"{item.percentile:.0f}"
                    if item.percentile == item.percentile
                    else "—",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_municipality_panel(
    mun_prices: pd.DataFrame,
    postal_code: str,
    boundaries: dict,
    quarter: str,
    building_type_code: str | None,
) -> None:
    code = str(postal_code).zfill(5)
    muni_code = postal_to_municipality_codes(boundaries).get(code)
    if muni_code is None:
        st.warning("This area has no municipality code on the boundary data.")
        return
    card = municipality_map_card_data(
        mun_prices, muni_code, quarter, building_type_code
    )
    if card is None:
        st.warning("No municipality figures for this selection.")
        return

    st.subheader("Municipality figure")
    st.markdown(f"**{card.municipality_name}** ({card.municipality_code})")
    st.caption(
        f"Annual figures for **{card.year}** (Statistics Finland municipality table). "
        "You clicked a postal-code area without its own published price; "
        "the map colour comes from the municipality average."
    )

    c1, c2 = st.columns(2)
    c1.metric(
        "Price per m² (EUR)",
        format_metric_value(card.price_per_sqm, METRIC_PRICE)
        if card.price_per_sqm is not None
        else "No data",
    )
    c2.metric(
        "1-year change",
        format_metric_value(card.pct_change_1y, METRIC_CHANGE_1Y)
        if card.pct_change_1y is not None
        else "—",
    )
    c3, c4 = st.columns(2)
    c3.metric(
        "5-year change",
        format_metric_value(card.pct_change_5y, METRIC_CHANGE_5Y)
        if card.pct_change_5y is not None
        else "—",
    )
    sales_text = "—"
    if card.transactions is not None:
        sales_text = str(card.transactions)
    c4.metric(f"Sales ({card.year})", sales_text)


def _render_detail_panel(
    prices: pd.DataFrame,
    boundaries: dict,
    cpi: pd.DataFrame,
    demographics: pd.DataFrame,
    national_demographics: pd.Series,
    postal_code: str,
    quarter: str,
    map_building_type_code: str | None,
) -> None:
    code = str(postal_code).zfill(5)
    st.subheader("Area detail")

    bt_options = [c for c, _ in building_type_panel_options()]
    default_bt = (
        map_building_type_code
        if map_building_type_code in bt_options
        else "all"
    )
    panel_bt_code = st.selectbox(
        "Building type (detail panel)",
        options=bt_options,
        format_func=lambda c: dict(building_type_panel_options())[c],
        index=bt_options.index(default_bt),
        key=f"panel_bt_{code}",
    )
    panel_bt_label = resolve_panel_building_type(prices, panel_bt_code)
    price_mode = st.radio(
        "Trend chart prices",
        options=("nominal", "real"),
        format_func=lambda v: "Nominal" if v == "nominal" else "Real (inflation-adjusted)",
        horizontal=True,
        key=f"panel_price_mode_{code}",
    )
    index_mode = st.checkbox("Index to 100 at the start", key=f"panel_index_{code}")

    summary = summarize_area(prices, code, quarter, building_type=panel_bt_label)
    area_name = area_display_name(prices, code) or "—"
    municipality = municipality_name_from_prices(prices, code)
    bt_display = _format_building_type_display(prices, panel_bt_code)

    st.markdown(
        format_area_header(code, area_name if area_name != "—" else "", municipality)
    )
    st.caption(f"Quarter **{quarter}** · {bt_display}")

    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    sales_4q = trailing_sales_count(prices, code, quarter, panel_bt_code)
    rel_text = reliability_explanation(summary.get("reliability"), sales_4q)

    price = summary["price_per_sqm"]
    yoy = summary["pct_change_1y"]
    five = summary["pct_change_5y"]
    rank = summary.get("rank")
    pct = summary.get("percentile")

    c1, c2 = st.columns(2)
    c1.metric(
        "Price per m² (EUR)",
        f"{price:,.0f}" if price == price else "No data",
        help="Nominal price per square metre for the selected quarter and building type.",
    )
    c2.metric(
        "1-year change",
        f"{yoy:+.1f}%" if yoy == yoy else "—",
        help="Change in price per m² versus the same quarter one year earlier.",
    )

    c3, c4 = st.columns(2)
    c3.metric(
        "5-year change",
        f"{five:+.1f}%" if five == five else "—",
        help="Change in price per m² versus the same quarter five years earlier.",
    )
    c4.metric(
        "Sales (4 quarters)",
        str(int(round(sales_4q))) if sales_4q == sales_4q else "—",
        help="Sales in the last four quarters.",
    )

    market_act = market_activity_for_area(
        prices, code, quarter, panel_bt_code, demographics
    )

    c5, c6 = st.columns(2)
    c5.metric(
        "Rank",
        str(int(rank)) if pd.notna(rank) else "—",
        help="Rank among all areas in this quarter; 1 = most expensive.",
    )
    c6.metric(
        "Percentile",
        f"{pct:.0f}th" if pct == pct else "—",
        help="Percentile among all areas in this quarter (higher = more expensive).",
    )

    c7, c8 = st.columns(2)
    c7.metric(
        "Market activity",
        f"{market_act:.1f} per 1,000 inh."
        if market_act == market_act
        else "—",
        help=MARKET_ACTIVITY_METRIC_HELP,
    )
    c8.empty()
    market_act_caption = (
        market_activity_panel_caption(sales_4q) if market_act == market_act else None
    )
    if market_act_caption:
        st.caption(market_act_caption)

    st.caption(rel_text)

    demo_index = demographics.set_index(
        demographics["postal_code"].astype(str).str.zfill(5)
    )
    if code in demo_index.index:
        area_demo = demo_index.loc[code]
        profile_items = build_area_profile_items(area_demo, national_demographics)
        profile_year = area_profile_data_year(area_demo, national_demographics)
        with st.expander("Area profile"):
            if profile_year is not None:
                st.caption(f"Statistics Finland Paavo, {profile_year}.")
            for item in profile_items:
                if item.help:
                    st.markdown(
                        f"- {item.label} "
                        f'<span title="{item.help}">—</span> {item.national_text}',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"- {item.label} {item.area_text} {item.national_text}"
                    )

    if st.button("Add to comparison", key=f"add_compare_{code}"):
        if _add_to_compare(code):
            st.success(f"{code} added to comparison.")

    trend_data = _cached_trend_chart_data(
        prices,
        boundaries,
        cpi,
        code,
        panel_bt_label,
        index_mode,
        price_mode == "real",
    )
    trend_fig = build_trend_figure(trend_data, area_label=f"{code} {area_name}".strip())
    st.plotly_chart(trend_fig, use_container_width=True, key=f"trend_{code}")
    if price_mode == "real":
        st.caption(
            "Real prices remove general inflation using Statistics Finland's consumer "
            "price index, so the trend shows purchasing-power change rather than "
            "nominal euro amounts."
        )

    if not trend_data.municipality_available:
        st.caption(
            "Municipality comparison line is not shown because municipality metadata "
            "is missing from the boundary data for this area."
        )

    area_series = _cached_area_prices(prices, code, panel_bt_label)
    unusual = flag_unusual_quarter_changes(area_series)
    if unusual:
        st.markdown("**Unusual quarter-on-quarter movements**")
        for item in unusual:
            st.write(
                f"- **{item['quarter']}**: {item['pct_change_qoq']:+.1f}% change "
                f"(typical spread ±{item['threshold_qoq']:.1f}% for this area)"
            )
        st.caption(
            "Small sample sizes can produce sharp spikes; treat flagged quarters with caution."
        )

    sales_counts = _cached_transaction_counts(prices, code, panel_bt_label)
    if not sales_counts.empty:
        st.plotly_chart(
            build_sales_volume_figure(sales_counts),
            use_container_width=True,
            key=f"sales_{code}",
        )
        st.caption(
            "Transaction counts are shown from 2020 onward, when public counts are available."
        )

    export_df = area_detail_export_frame(prices, code, panel_bt_label)
    st.download_button(
        "Download area data (CSV)",
        data=export_df.to_csv(index=False),
        file_name=f"housing_{code}.csv",
        mime="text/csv",
        key=f"download_{code}",
    )


try:
    with st.spinner("Loading housing prices and map boundaries…"):
        prices, boundaries, cpi, demographics, manifest = _load_housing_data()
        national_demographics = _load_national_demographics()
        mun_prices, mun_boundaries = _load_municipality_data()
except Exception as exc:  # noqa: BLE001 — show reason in UI
    st.error(_friendly_load_error(exc))
    st.stop()

if prices.empty or not (boundaries.get("features")):
    st.error("Housing data loaded but appears empty. Try again later or use fixtures.")
    st.stop()

with st.sidebar:
    st.header("Map controls")
    quarters = list_quarters(prices)
    default_quarter = default_map_quarter(prices) or quarters[-1]
    quarter = st.selectbox(
        "Quarter",
        options=quarters,
        index=quarters.index(default_quarter) if default_quarter in quarters else len(quarters) - 1,
        help=(
            "Defaults to the latest quarter with enough published prices across all "
            "building types. A specific building type may still show as provisional "
            "for this quarter if its own coverage is low."
        ),
    )
    building_type_code = st.selectbox(
        "Building type",
        options=[code for code, _ in BUILDING_TYPE_CHOICES],
        format_func=lambda c: dict(BUILDING_TYPE_CHOICES)[c],
        index=0,
    )
    metric = st.selectbox(
        "Metric layer",
        options=[key for key, _ in METRIC_CHOICES],
        format_func=lambda k: dict(METRIC_CHOICES)[k],
        index=0,
    )
    use_full_color_range = st.checkbox(
        "Use the full value range",
        value=False,
        help="When off, the colour scale uses the 2nd–98th percentile so outliers do not wash out the map.",
    )
    municipality_data_ready = mun_prices is not None and mun_boundaries is not None
    fill_gaps_with_municipality = st.checkbox(
        "Fill gaps with municipality values",
        value=True,
        disabled=not municipality_data_ready,
        help=(
            "Colour postal-code areas without their own price using yearly municipality "
            "averages underneath."
        ),
    )
    afford_inputs = _affordability_sidebar_inputs()

boundary_edition = None
if manifest:
    boundaries_meta = (manifest.get("files") or {}).get("boundaries.geojson.gz") or {}
    if isinstance(boundaries_meta, dict):
        boundary_edition = boundaries_meta.get("boundary_edition")

_init_compare_session_state()

max_affordable_price = _compute_max_affordable(afford_inputs)

map_tab, afford_tab, compare_tab, relationships_tab, my_home_tab = st.tabs(
    ["Map", "Affordability", "Compare", "Relationships", "My home"]
)

with map_tab:
    map_df = _cached_map_frame(
        prices,
        boundaries,
        cpi,
        demographics,
        quarter,
        building_type_code,
        metric,
        afford_inputs["size_sqm"],
        max_affordable_price,
        afford_inputs["annual_rate_pct"],
        afford_inputs["loan_years"],
        afford_inputs["down_payment_value"],
        afford_inputs["down_mode"] == "percent",
    )
    hybrid_active = (
        fill_gaps_with_municipality
        and municipality_data_ready
        and hybrid_metric_supported(metric)
    )
    mun_map_df = pd.DataFrame()
    municipality_year: int | None = None
    if hybrid_active and mun_prices is not None:
        mun_map_df, municipality_year = prepare_municipality_map_dataframe(
            mun_prices,
            cpi,
            quarter,
            building_type_code,
            metric,
            size_sqm=afford_inputs["size_sqm"],
            max_affordable_price=max_affordable_price,
        )
        if mun_map_df.empty:
            hybrid_active = False
        else:
            p2m = postal_to_municipality_codes(boundaries)
            map_df = classify_hybrid_postal_coverage(
                map_df, metric, p2m, mun_map_df
            )

    color_range = None
    if metric != METRIC_FITS_BUDGET and not map_df.empty:
        if hybrid_active and not mun_map_df.empty:
            color_range = hybrid_metric_color_range(
                map_df,
                mun_map_df,
                metric,
                use_full_range=use_full_color_range,
            )
        else:
            color_range = metric_color_range(
                map_df.loc[~map_df["missing"], metric],
                metric,
                use_full_range=use_full_color_range,
            )

    if hybrid_active and mun_boundaries is not None:
        fig = build_hybrid_choropleth_figure(
            map_df,
            mun_map_df,
            boundaries,
            mun_boundaries,
            metric,
            color_range=color_range,
            fill_gaps=True,
        )
    else:
        fig = build_choropleth_figure(
            map_df, boundaries, metric, color_range=color_range
        )

    if boundary_edition:
        st.caption(f"Map boundaries: {boundary_edition} (Statistics Finland).")

    with st.expander("How to read the map"):
        if metric == METRIC_FITS_BUDGET:
            st.markdown(
                "- **Green** — typical price for your size fits within the maximum affordable "
                f"price ({max_affordable_price:,.0f} EUR from the sidebar budget).\n"
                "- **Amber** — typical price is up to 20% above that maximum.\n"
                "- **Red** — more than 20% above the maximum.\n"
                "- **Grey** — no published price for the chosen quarter and building type "
                "(budget layer only; value layers leave these areas unfilled).\n"
                "- Typical price = area EUR/m² × apartment size (sidebar). "
                "Uses the same loan assumptions as the Affordability tab."
            )
            for key, label in BUDGET_FIT_LABELS.items():
                st.caption(f"**{label}**")
        else:
            bullets = (
                "- **Darker fill** means a **higher** value; **lighter fill** means lower (light grey to black).\n"
                "- **Coloured areas** show the selected metric for the chosen quarter and building type.\n"
                "- **Unfilled areas** (outline only) have no published value for that selection "
                f"({METRIC_PRICE.replace('_', ' ')} missing or suppressed); they are never shown as zero.\n"
                "- With **Fill gaps with municipality values** on, a **lighter municipality fill** "
                "shows yearly averages where postal-code prices are missing; postal detail stays on top.\n"
                "- **Amber borders** and the hover note *Based on few sales* mark **low reliability** "
                "(fewer than ten sales in the last four quarters).\n"
                "- Hover a region for postal code, area name, metric value, sales, and reliability."
            )
            if metric == METRIC_MARKET_ACTIVITY:
                bullets += (
                    "\n- **Market activity** uses transaction counts (from 2020 onward) for old "
                    "dwellings in housing companies and the postal area's total population (Paavo); "
                    "it is a rough activity index, not a turnover rate of the housing stock."
                )
            st.markdown(bullets)

    if not municipality_data_ready:
        st.caption(
            "Municipality fallback data is not in this checkout, so the map uses postal-code "
            "prices only (same as before)."
        )
    elif hybrid_active and municipality_year is not None:
        st.caption(
            f"Municipality values are annual figures for {municipality_year}; "
            f"postal-code values are for {quarter}."
        )
        bt_caption = building_type_mapping_caption(building_type_code)
        if bt_caption:
            st.caption(bt_caption)
    elif not hybrid_metric_supported(metric):
        reason = hybrid_unsupported_reason(metric)
        if reason:
            st.caption(reason)

    if "selected_postal_code" not in st.session_state:
        st.session_state.selected_postal_code = default_selected_postal_code(prices)
    if "selected_map_level" not in st.session_state:
        st.session_state.selected_map_level = "postal"

    map_col, detail_col = st.columns([5, 4])

    with map_col:
        search_query = st.text_input(
            "Search by postal code or area name",
            placeholder="e.g. 00100 or Punavuori",
        )

        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            key="housing_map",
            config=plotly_map_chart_config(),
        )

        if not map_df.empty:
            if hybrid_active and "coverage" in map_df.columns:
                counts = hybrid_coverage_counts(map_df)
                st.caption(
                    f"{counts.own_postal:,} postal-code areas have their own price; "
                    f"{counts.municipality_coloured:,} more are coloured from municipality "
                    f"averages; {counts.no_data:,} have no data."
                )
            else:
                areas_with_metric = int((~map_df["missing"]).sum())
                total_areas = len(map_df)
                st.caption(
                    f"{areas_with_metric} of {total_areas} areas have a published price for this "
                    "selection. Statistics Finland publishes prices only for areas with enough "
                    "sales; the rest are unfilled (outline only) and are never treated as zero."
                )
            if metric != METRIC_FITS_BUDGET and not use_full_color_range:
                if metric == METRIC_PRICE:
                    st.caption(
                        "Colour scale is fixed at 0 to 4,000 EUR/m²; more expensive areas "
                        "use the darkest colour"
                    )
                else:
                    st.caption(
                        "Colour scale covers the 2nd to 98th percentile; more extreme areas "
                        "use the end colours."
                    )
            typical_coverage = typical_quarter_price_coverage(
                prices, quarter, building_type_code
            )
            if typical_coverage is not None and not quarter_meets_coverage_threshold(
                prices, quarter, building_type_code, typical=typical_coverage
            ):
                price_areas = count_areas_with_published_price(
                    prices, quarter, building_type_code
                )
                typical_rounded = int(round(typical_coverage))
                st.caption(
                    f"{quarter} is provisional: only {price_areas} areas have a published "
                    f"price (typically about {typical_rounded})."
                )

        map_pick = map_selection_from_event_or_postal(selection)
        clicked_code = map_pick.postal_code if map_pick else None
        clicked_level = map_pick.level if map_pick else "postal"

        search_hits = search_area_matches(map_df, search_query)
        if search_hits:
            picked = st.selectbox(
                "Matching areas",
                options=search_hits,
                format_func=lambda c: f"{c} — {map_df.loc[map_df['postal_code']==c, 'area_name'].iloc[0]}",
            )
            if st.button("Show selected area"):
                st.session_state.selected_postal_code = picked
                st.session_state.selected_map_level = "postal"
        elif search_query.strip():
            st.caption("No areas match your search.")

        if clicked_code:
            st.session_state.selected_postal_code = clicked_code
            st.session_state.selected_map_level = clicked_level

    with detail_col:
        selected = st.session_state.selected_postal_code
        if selected:
            if (
                st.session_state.get("selected_map_level") == "municipality"
                and hybrid_active
                and mun_prices is not None
            ):
                _render_municipality_panel(
                    mun_prices,
                    selected,
                    boundaries,
                    quarter,
                    building_type_code,
                )
            else:
                _render_detail_panel(
                    prices,
                    boundaries,
                    cpi,
                    demographics,
                    national_demographics,
                    selected,
                    quarter,
                    building_type_code,
                )
        else:
            st.info(
                "Click a map area or search and choose **Show selected area** to open the detail panel."
            )

with afford_tab:
    _render_affordability_tab(
        prices,
        quarter,
        building_type_code,
        afford_inputs,
        st.session_state.get("selected_postal_code"),
    )

with compare_tab:
    _render_compare_tab(prices, quarter, building_type_code, cpi, demographics)

with relationships_tab:
    _render_relationships_tab(
        prices, quarter, building_type_code, cpi, demographics
    )

with my_home_tab:
    _render_my_home_tab(
        prices,
        cpi,
        st.session_state.get("selected_postal_code"),
    )

st.divider()
footer_parts = [
    "Data: Statistics Finland — "
    "[Prices per square metre by postal code (PxWeb)](https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/ashi/13mt.px), "
    "[Paavo open data by postal code](https://pxdata.stat.fi/PxWeb/api/v1/en/Postinumeroalueittainen_avoin_tieto/), "
    "and [postal-code boundaries (WFS)](https://geo.stat.fi/geoserver/postialue/wfs)."
]
if manifest and manifest.get("fetch_date"):
    footer_parts.append(f" Snapshot fetched {manifest['fetch_date']}.")
st.caption("".join(footer_parts))
