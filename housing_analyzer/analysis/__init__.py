"""Pure analysis helpers for housing price tables."""

from housing_analyzer.analysis.compare import (
    area_catalog,
    build_compare_chart_data,
    build_compare_figure,
    build_comparison_table,
    search_area_catalog,
    summaries_for_compare,
)
from housing_analyzer.analysis.similar_areas import (
    similar_areas,
    similar_areas_explanation,
)
from housing_analyzer.analysis.metrics import (
    area_prices_at,
    pct_change,
    quarter_index,
    rank_percentile,
    real_pct_change,
    regional_average,
    reliability,
    reliability_at,
    shift_quarter,
    summarize_area,
    summarize_areas,
    to_real,
)

__all__ = [
    "area_catalog",
    "area_prices_at",
    "build_compare_chart_data",
    "build_compare_figure",
    "build_comparison_table",
    "pct_change",
    "quarter_index",
    "real_pct_change",
    "reliability",
    "reliability_at",
    "rank_percentile",
    "regional_average",
    "search_area_catalog",
    "shift_quarter",
    "similar_areas",
    "similar_areas_explanation",
    "summaries_for_compare",
    "summarize_area",
    "summarize_areas",
    "to_real",
]
