"""Price–demographics relationships: price-to-income ratio and scatter plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

CORRELATION_DISCLAIMER = (
    "Correlation describes how two measures move together across areas; it does not "
    "show that one causes the other."
)


def price_to_income_ratio(
    price_per_sqm: float,
    median_income_eur: float,
) -> float:
    """Rough affordability indicator: latest EUR/m² divided by median disposable income (EUR/year).

    This compares a per-square-metre dwelling price to a per-person income series from Paavo.
    It is not a formal housing-cost ratio and ignores dwelling size, household composition,
    and taxes; use it only as a coarse map overlay.
    """
    if (
        price_per_sqm is None
        or median_income_eur is None
        or (isinstance(price_per_sqm, float) and np.isnan(price_per_sqm))
        or (isinstance(median_income_eur, float) and np.isnan(median_income_eur))
        or pd.isna(price_per_sqm)
        or pd.isna(median_income_eur)
    ):
        return float("nan")
    income = float(median_income_eur)
    if income == 0.0:
        return float("nan")
    return float(price_per_sqm) / income


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson r for paired finite samples; returns NaN when undefined."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    xs = xs[mask]
    ys = ys[mask]
    if np.std(xs, ddof=1) == 0.0 or np.std(ys, ddof=1) == 0.0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


@dataclass(frozen=True)
class RelationshipPlotSpec:
    key: str
    title: str
    x_label: str
    y_label: str
    x_column: str


RELATIONSHIP_PLOTS: tuple[RelationshipPlotSpec, ...] = (
    RelationshipPlotSpec(
        key="income",
        title="Price per m² vs median income",
        x_label="Median disposable income (EUR/year)",
        y_label="Price per m² (EUR)",
        x_column="median_income_eur",
    ),
    RelationshipPlotSpec(
        key="age_65",
        title="Price per m² vs share aged 65+",
        x_label="Share of population aged 65+",
        y_label="Price per m² (EUR)",
        x_column="share_age_65_plus",
    ),
    RelationshipPlotSpec(
        key="higher_ed",
        title="Price per m² vs higher-education share",
        x_label="Share with higher university degree (18+)",
        y_label="Price per m² (EUR)",
        x_column="share_higher_education",
    ),
    RelationshipPlotSpec(
        key="unemployment",
        title="Price per m² vs unemployment rate",
        x_label="Unemployment rate (labour force)",
        y_label="Price per m² (EUR)",
        x_column="unemployment_rate",
    ),
    RelationshipPlotSpec(
        key="rented",
        title="Price per m² vs rented-household share",
        x_label="Share of rented households",
        y_label="Price per m² (EUR)",
        x_column="rented_share",
    ),
)


@dataclass(frozen=True)
class RelationshipPlotResult:
    spec: RelationshipPlotSpec
    n_areas: int
    correlation: float
    figure: go.Figure


@dataclass(frozen=True)
class RelationshipsSummary:
    included: pd.DataFrame
    excluded_count: int
    plots: tuple[RelationshipPlotResult, ...]


def _reliable_price_row(row: Mapping[str, Any]) -> bool:
    rel = row.get("reliability")
    price = row.get("price_per_sqm")
    if rel != "ok":
        return False
    if price is None or (isinstance(price, float) and np.isnan(price)) or pd.isna(price):
        return False
    return True


def prepare_relationships_frame(
    summaries: pd.DataFrame,
    demographics: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Areas with OK price reliability and complete demographics for relationship plots."""
    demo = demographics.set_index(demographics["postal_code"].astype(str).str.zfill(5))
    rows: list[dict[str, Any]] = []
    excluded = 0
    for code, row in summaries.iterrows():
        postal = str(code).zfill(5)
        if not _reliable_price_row(row):
            excluded += 1
            continue
        if postal not in demo.index:
            excluded += 1
            continue
        demo_row = demo.loc[postal]
        needed = (
            "median_income_eur",
            "share_age_65_plus",
            "share_higher_education",
        )
        if any(pd.isna(demo_row.get(col)) for col in needed):
            excluded += 1
            continue
        unemployment_rate = demo_row["unemployment_rate"]
        rented_share = demo_row["rented_share"]
        rows.append(
            {
                "postal_code": postal,
                "price_per_sqm": float(row["price_per_sqm"]),
                "median_income_eur": float(demo_row["median_income_eur"]),
                "share_age_65_plus": float(demo_row["share_age_65_plus"]),
                "share_higher_education": float(demo_row["share_higher_education"]),
                "unemployment_rate": float(unemployment_rate)
                if pd.notna(unemployment_rate)
                else float("nan"),
                "rented_share": float(rented_share) if pd.notna(rented_share) else float("nan"),
            }
        )
    return pd.DataFrame(rows), excluded


def build_scatter_figure(
    frame: pd.DataFrame,
    spec: RelationshipPlotSpec,
    *,
    correlation: float,
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=frame[spec.x_column],
            y=frame["price_per_sqm"],
            mode="markers",
            marker={"size": 7, "opacity": 0.65},
            text=frame["postal_code"],
            hovertemplate=(
                "Postal code %{text}<br>"
                f"{spec.x_label}: %{{x:.4g}}<br>"
                f"{spec.y_label}: %{{y:,.0f}}<extra></extra>"
            ),
        )
    )
    r_text = f"{correlation:.2f}" if correlation == correlation else "—"
    fig.update_layout(
        title=f"{spec.title} (n={len(frame)}, r={r_text})",
        xaxis_title=spec.x_label,
        yaxis_title=spec.y_label,
        height=360,
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    return fig


def build_relationships_summary(
    summaries: pd.DataFrame,
    demographics: pd.DataFrame,
) -> RelationshipsSummary:
    frame, excluded = prepare_relationships_frame(summaries, demographics)
    plots: list[RelationshipPlotResult] = []
    for spec in RELATIONSHIP_PLOTS:
        if frame.empty:
            corr = float("nan")
            plot_frame = frame
        else:
            plot_frame = frame.dropna(subset=[spec.x_column, "price_per_sqm"])
            corr = pearson_correlation(
                plot_frame[spec.x_column].tolist(),
                plot_frame["price_per_sqm"].tolist(),
            )
        plots.append(
            RelationshipPlotResult(
                spec=spec,
                n_areas=len(plot_frame),
                correlation=corr,
                figure=build_scatter_figure(plot_frame, spec, correlation=corr),
            )
        )
    return RelationshipsSummary(
        included=frame,
        excluded_count=excluded,
        plots=tuple(plots),
    )
