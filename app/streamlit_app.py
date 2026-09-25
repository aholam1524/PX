"""Streamlit UI for the housing price analyzer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from housing_analyzer.analysis.metrics import summarize_area
from housing_analyzer.data import load_boundaries, load_manifest, load_prices
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

st.set_page_config(page_title="Housing price analyzer", layout="wide")

st.title("Housing price analyzer")
st.write(
    "Explore Finnish postal-code areas on a map coloured by housing price metrics, "
    "with trends and comparisons for a selected area."
)
st.caption("For information only — not investment advice.")


@st.cache_data(show_spinner=False)
def _load_housing_data() -> tuple:
    prices = load_prices()
    boundaries = load_boundaries()
    manifest = load_manifest()
    return prices, boundaries, manifest


@st.cache_data(show_spinner=False)
def _cached_map_frame(_prices, _boundaries, quarter, building_type_code, metric):
    """Map table for one selection. The large inputs are not hashed (leading underscore);
    the selection values are the cache key, so clicking the map does not recompute it."""
    return prepare_map_dataframe(_prices, _boundaries, quarter, building_type_code, metric)


def _friendly_load_error(exc: BaseException) -> str:
    return f"Could not load housing data: {exc}"


def _render_summary_card(
    prices,
    postal_code: str,
    quarter: str,
    building_type_code: str | None,
) -> None:
    bt_label = resolve_building_type_label(prices, building_type_code)
    summary = summarize_area(
        prices,
        postal_code,
        quarter,
        building_type=bt_label,
    )
    st.subheader(f"{summary['postal_code']} — selected area")
    col1, col2, col3 = st.columns(3)
    price = summary["price_per_sqm"]
    col1.metric(
        "Price per m²",
        f"{price:,.0f} EUR/m²" if price == price else "No data",
    )
    yoy = summary["pct_change_1y"]
    col2.metric(
        "1-year change",
        f"{yoy:+.1f}%" if yoy == yoy else "—",
    )
    five = summary["pct_change_5y"]
    col3.metric(
        "5-year change",
        f"{five:+.1f}%" if five == five else "—",
    )
    rel = summary.get("reliability")
    rank = summary.get("rank")
    pct = summary.get("percentile")
    rank_text = ""
    if pd.notna(rank) and pct == pct:
        rank_text = f" · Rank {int(rank)} · {pct:.0f}th percentile"
    st.caption(f"Reliability: {rel or 'unknown'}{rank_text}")


try:
    with st.spinner("Loading housing prices and map boundaries…"):
        prices, boundaries, manifest = _load_housing_data()
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

map_df = _cached_map_frame(prices, boundaries, quarter, building_type_code, metric)
fig = build_choropleth_figure(map_df, boundaries, metric)

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
    if st.button("Show selected area summary"):
        st.session_state.selected_postal_code = picked
elif search_query.strip():
    st.caption("No areas match your search.")

if clicked_code:
    st.session_state.selected_postal_code = clicked_code

selected = st.session_state.selected_postal_code
if selected:
    _render_summary_card(prices, selected, quarter, building_type_code)

st.divider()
footer_parts = [
    "Data: Statistics Finland — "
    "[Prices per square metre by postal code (PxWeb)](https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/ashi/13mt.px) "
    "and [postal-code boundaries (WFS)](https://geo.stat.fi/geoserver/postialue/wfs)."
]
if manifest and manifest.get("fetch_date"):
    footer_parts.append(f" Snapshot fetched {manifest['fetch_date']}.")
st.caption("".join(footer_parts))
