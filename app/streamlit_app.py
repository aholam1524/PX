"""Streamlit UI for the housing price analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

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
from housing_analyzer.data.demographics import attach_demographics_to_summaries
from housing_analyzer.data.cpi import cpi_by_quarter
from housing_analyzer.data.paths import use_fixtures
from housing_analyzer.data.snapshot import snapshot_is_complete
from housing_analyzer.affordability import (
    AFFORDABILITY_DISCLAIMER,
    down_payment_from_inputs,
    loan_amount,
    max_price,
    monthly_payment,
    total_interest,
    typical_dwelling_price,
)
from housing_analyzer.map import (
    BUDGET_FIT_LABELS,
    BUILDING_TYPE_CHOICES,
    METRIC_CHOICES,
    METRIC_FITS_BUDGET,
    METRIC_PRICE,
    build_choropleth_figure,
    latest_quarter_with_data,
    list_quarters,
    postal_code_from_selection,
    prepare_budget_fit_dataframe,
    prepare_map_dataframe,
    resolve_building_type_label,
    search_area_matches,
    trailing_sales_by_area,
)
from housing_analyzer.panel import (
    area_detail_export_frame,
    area_display_name,
    build_sales_volume_figure,
    build_trend_chart_data,
    build_trend_figure,
    building_type_panel_options,
    flag_unusual_quarter_changes,
    format_area_header,
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
    price_source = st.radio(
        "Price for calculator",
        options=("target", "area"),
        format_func=lambda v: "Target price" if v == "target" else "Use area prices",
        key="afford_price_source",
    )
    target_price = st.number_input(
        "Target price (EUR)",
        min_value=0.0,
        value=250_000.0,
        step=5_000.0,
        disabled=price_source != "target",
        key="afford_target_price",
    )
    return {
        "size_sqm": float(size_sqm),
        "monthly_budget": float(monthly_budget),
        "annual_rate_pct": float(annual_rate_pct),
        "loan_years": float(loan_years),
        "down_mode": down_mode,
        "down_payment_value": float(down_payment_value),
        "price_source": price_source,
        "target_price": float(target_price),
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


def _render_affordability_tab(
    prices: pd.DataFrame,
    quarter: str,
    building_type_code: str | None,
    afford: dict[str, float | bool | str],
    selected_postal_code: str | None,
) -> None:
    st.subheader("Affordability calculator")
    st.caption(
        f"Quarter **{quarter}** · {_format_building_type_display(prices, building_type_code)}"
    )
    st.info(AFFORDABILITY_DISCLAIMER)

    max_affordable = _compute_max_affordable(afford)
    st.metric("Maximum affordable price", f"{max_affordable:,.0f} EUR")
    if (
        afford["down_mode"] == "percent"
        and float(afford["down_payment_value"]) >= 100.0
    ):
        st.caption(
            "A 100%+ down payment covers any price from savings alone, so this isn't "
            "limited by your monthly budget."
        )

    use_percent = afford["down_mode"] == "percent"
    if afford["price_source"] == "target":
        calc_price = float(afford["target_price"])
    else:
        if not selected_postal_code:
            st.warning(
                "Select an area on the **Map** tab (or search there) to use its typical "
                f"price for {afford['size_sqm']:g} m²."
            )
            return
        code = str(selected_postal_code).zfill(5)
        bt_label = resolve_building_type_label(prices, building_type_code)
        summary = summarize_area(prices, code, quarter, building_type=bt_label)
        price_sqm = summary.get("price_per_sqm")
        if price_sqm != price_sqm or pd.isna(price_sqm):
            st.warning(f"No published price for {code} in this selection.")
            return
        calc_price = typical_dwelling_price(float(price_sqm), float(afford["size_sqm"]))
        st.caption(
            f"Using typical price for **{code}** ({calc_price:,.0f} EUR for "
            f"{afford['size_sqm']:g} m² at {price_sqm:,.0f} EUR/m²)."
        )

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
        return

    c1, c2 = st.columns(2)
    c1.metric("Monthly payment", f"{payment:,.2f} EUR")
    c2.metric("Total interest over term", f"{interest:,.0f} EUR")
    st.caption(
        f"For price **{calc_price:,.0f} EUR** with down payment **{down:,.0f} EUR** "
        f"and loan **{principal:,.0f} EUR**."
    )


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


def _render_detail_panel(
    prices: pd.DataFrame,
    boundaries: dict,
    cpi: pd.DataFrame,
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
    st.caption(rel_text)

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
except Exception as exc:  # noqa: BLE001 — show reason in UI
    st.error(_friendly_load_error(exc))
    st.stop()

if prices.empty or not (boundaries.get("features")):
    st.error("Housing data loaded but appears empty. Try again later or use fixtures.")
    st.stop()

with st.sidebar:
    st.header("Map controls")
    quarters = list_quarters(prices)
    default_quarter = latest_quarter_with_data(prices) or quarters[-1]
    quarter = st.selectbox(
        "Quarter",
        options=quarters,
        index=quarters.index(default_quarter) if default_quarter in quarters else len(quarters) - 1,
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
    afford_inputs = _affordability_sidebar_inputs()

boundary_edition = None
if manifest:
    boundaries_meta = (manifest.get("files") or {}).get("boundaries.geojson.gz") or {}
    if isinstance(boundaries_meta, dict):
        boundary_edition = boundaries_meta.get("boundary_edition")

_init_compare_session_state()

max_affordable_price = _compute_max_affordable(afford_inputs)

map_tab, afford_tab, compare_tab, relationships_tab = st.tabs(
    ["Map", "Affordability", "Compare", "Relationships"]
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
    )
    fig = build_choropleth_figure(map_df, boundaries, metric)

    if boundary_edition:
        st.caption(f"Map boundaries: {boundary_edition} (Statistics Finland).")

    with st.expander("How to read the map"):
        if metric == METRIC_FITS_BUDGET:
            st.markdown(
                "- **Green** — typical price for your size fits within the maximum affordable "
                f"price ({max_affordable_price:,.0f} EUR from the sidebar budget).\n"
                "- **Amber** — typical price is up to 20% above that maximum.\n"
                "- **Red** — more than 20% above the maximum.\n"
                "- **Grey** — no published price for the chosen quarter and building type.\n"
                "- Typical price = area EUR/m² × apartment size (sidebar). "
                "Uses the same loan assumptions as the Affordability tab."
            )
            for key, label in BUDGET_FIT_LABELS.items():
                st.caption(f"**{label}**")
        else:
            st.markdown(
                "- **Coloured areas** show the selected metric for the chosen quarter and building type.\n"
                "- **Grey areas** have no published price for that selection "
                f"({METRIC_PRICE.replace('_', ' ')} missing or suppressed); they are never shown as zero.\n"
                "- **Lighter fill** and the hover note *Based on few sales* mark **low reliability** "
                "(fewer than ten sales in the last four quarters).\n"
                "- Hover a region for postal code, area name, metric value, sales, and reliability."
            )

    if "selected_postal_code" not in st.session_state:
        st.session_state.selected_postal_code = None

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
        )

        clicked_code = postal_code_from_selection(selection)

        search_hits = search_area_matches(map_df, search_query)
        if search_hits:
            picked = st.selectbox(
                "Matching areas",
                options=search_hits,
                format_func=lambda c: f"{c} — {map_df.loc[map_df['postal_code']==c, 'area_name'].iloc[0]}",
            )
            if st.button("Show selected area"):
                st.session_state.selected_postal_code = picked
        elif search_query.strip():
            st.caption("No areas match your search.")

        if clicked_code:
            st.session_state.selected_postal_code = clicked_code

    with detail_col:
        selected = st.session_state.selected_postal_code
        if selected:
            _render_detail_panel(
                prices,
                boundaries,
                cpi,
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
