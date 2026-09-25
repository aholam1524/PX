"""Streamlit UI for the housing price analyzer."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Housing price analyzer", layout="wide")

st.title("Housing price analyzer")
st.write(
    "Explore Finnish postal-code areas on a map coloured by housing price metrics, "
    "with trends and comparisons for a selected area."
)
st.caption("For information only — not investment advice.")

st.subheader("Map (sample data)")
st.caption(
    "Placeholder map using sample points in Finland. Real postal-code data will be added in later tickets."
)

# Sample points: Helsinki, Tampere, Turku, Oulu
sample = pd.DataFrame(
    {
        "lat": [60.1699, 61.4978, 60.4518, 65.0121],
        "lon": [24.9384, 23.7610, 22.2666, 25.4651],
        "label": ["Helsinki (sample)", "Tampere (sample)", "Turku (sample)", "Oulu (sample)"],
    }
)
st.map(sample, latitude="lat", longitude="lon")
