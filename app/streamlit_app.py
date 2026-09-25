"""Streamlit UI for the housing price analyzer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from housing_analyzer.analysis.metrics import summarize_area
from housing_analyzer.data import load_boundaries, load_cpi, load_manifest, load_prices
from housing_analyzer.data.paths import use_fixtures
from housing_analyzer.data.snapshot import snapshot_is_complete
from housing_analyzer.map import (
    BUILDING_TYPE_CHOICES,
    METRIC_CHOICES,
    METRIC_PRICE,
    build_choropleth_figure,
    latest_quarter_with_data,
    list_quarters,
    postal_code_from_selection,
    prepare_map_dataframe,
    resolve_building_type_label,
    search_area_matches,
)
from housing_analyzer.panel import (
    area_detail_export_frame,
    area_display_name,
    build_sales_volume_figure,
    build_trend_chart_data,
    build_trend_figure,
    building_type_panel_options,
    flag_unusual_quarter_changes,
    municipality_name_from_prices,
    quarterly_area_prices,
    quarterly_transaction_counts,
    reliability_explanation,
    resolve_panel_building_type,
    trailing_sales_count,
)

st.set_page_config(page_title="Housing price analyzer", layout="wide")

st.title("Housing price analyzer")
st.write(
    "Explore Finnish postal-code areas on a map coloured by housing price metrics, "
    "with trends and comparisons for a selected area."
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
    manifest = load_manifest()
    return prices, boundaries, cpi, manifest


@st.cache_data(show_spinner=False)
def _cached_map_frame(_prices, _boundaries, _cpi, quarter, building_type_code, metric):
    """Map table for one selection. The large inputs are not hashed (leading underscore);
    the selection values are the cache key, so clicking the map does not recompute it."""
    return prepare_map_dataframe(
        _prices, _boundaries, quarter, building_type_code, metric, cpi_df=_cpi
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

    header_bits = [f"**{code}**", area_name]
    if municipality:
        header_bits.append(f"({municipality})")
    st.markdown(" · ".join(header_bits))
    st.caption(f"Quarter **{quarter}** · {bt_display}")

    sales_4q = trailing_sales_count(prices, code, quarter, panel_bt_code)
    rel_text = reliability_explanation(summary.get("reliability"), sales_4q)

    c1, c2, c3 = st.columns(3)
    price = summary["price_per_sqm"]
    c1.metric(
        "Price per m²",
        f"{price:,.0f} EUR/m²" if price == price else "No data",
    )
    yoy = summary["pct_change_1y"]
    c2.metric(
        "1-year change",
        f"{yoy:+.1f}%" if yoy == yoy else "—",
    )
    five = summary["pct_change_5y"]
    c3.metric(
        "5-year change",
        f"{five:+.1f}%" if five == five else "—",
    )

    c4, c5, c6 = st.columns(3)
    c4.metric(
        "Sales (last 4 quarters)",
        str(int(round(sales_4q))) if sales_4q == sales_4q else "—",
    )
    rank = summary.get("rank")
    pct = summary.get("percentile")
    c5.metric(
        "Rank among areas",
        str(int(rank)) if pd.notna(rank) else "—",
    )
    c6.metric(
        "Percentile",
        f"{pct:.0f}th" if pct == pct else "—",
    )
    st.caption(rel_text)

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
        prices, boundaries, cpi, manifest = _load_housing_data()
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

boundary_edition = None
if manifest:
    boundaries_meta = (manifest.get("files") or {}).get("boundaries.geojson.gz") or {}
    if isinstance(boundaries_meta, dict):
        boundary_edition = boundaries_meta.get("boundary_edition")

map_df = _cached_map_frame(prices, boundaries, cpi, quarter, building_type_code, metric)
fig = build_choropleth_figure(map_df, boundaries, metric)

if boundary_edition:
    st.caption(f"Map boundaries: {boundary_edition} (Statistics Finland).")

with st.expander("How to read the map"):
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

map_col, detail_col = st.columns([3, 2])

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
        st.info("Click a map area or search and choose **Show selected area** to open the detail panel.")

st.divider()
footer_parts = [
    "Data: Statistics Finland — "
    "[Prices per square metre by postal code (PxWeb)](https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/ashi/13mt.px) "
    "and [postal-code boundaries (WFS)](https://geo.stat.fi/geoserver/postialue/wfs)."
]
if manifest and manifest.get("fetch_date"):
    footer_parts.append(f" Snapshot fetched {manifest['fetch_date']}.")
st.caption("".join(footer_parts))
