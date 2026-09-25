"""Find postal-code areas most similar to a target on standardized price features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# Feature column names and weights in one place for future income/demographics tickets.
SIMILARITY_FEATURES: tuple[tuple[str, float], ...] = (
    ("price_per_sqm", 1.0),
    ("pct_change_5y", 1.0),
)

SIMILARITY_METHOD_SENTENCE = (
    "Similar areas are the five closest among areas with OK reliability, measured "
    "with weighted Euclidean distance on z-scored price per m² and 5-year change "
    "(each feature centered and scaled across the eligible pool)."
)


@dataclass(frozen=True)
class SimilarArea:
    postal_code: str
    distance: float
    price_per_sqm: float
    pct_change_5y: float
    pct_change_1y: float
    rank: Any
    percentile: float


def _normalize_code(postal_code: str) -> str:
    return str(postal_code).zfill(5)


def _feature_columns() -> tuple[str, ...]:
    return tuple(name for name, _ in SIMILARITY_FEATURES)


def _weighted_distance(
    target: np.ndarray, candidate: np.ndarray, weights: np.ndarray
) -> float:
    diff = target - candidate
    return float(np.sqrt(np.sum(weights * diff * diff)))


def similar_areas(
    summaries: pd.DataFrame,
    target_postal_code: str,
    *,
    reliability_column: str = "reliability",
    limit: int = 5,
) -> list[SimilarArea]:
    """Return up to ``limit`` areas most similar to ``target_postal_code``.

    Only areas with reliability ``"ok"`` and non-missing similarity features are
    considered. The target itself is never returned. Ties on distance break on
    ascending postal code.
    """
    target = _normalize_code(target_postal_code)
    if summaries.empty or target not in summaries.index:
        return []

    features = _feature_columns()
    weights = np.array([w for _, w in SIMILARITY_FEATURES], dtype=float)

    pool = summaries.loc[summaries[reliability_column] == "ok"].copy()
    if pool.empty or target not in pool.index:
        return []

    complete = pool.dropna(subset=list(features))
    if target not in complete.index:
        return []

    values = complete[list(features)].astype(float)
    means = values.mean(axis=0).to_numpy()
    stds = values.std(axis=0, ddof=0).to_numpy()
    stds = np.where(stds == 0.0, 1.0, stds)
    z = (values.to_numpy() - means) / stds

    z_index = {code: z[i] for i, code in enumerate(complete.index.astype(str))}
    target_z = z_index[target]

    candidates: list[tuple[float, str]] = []
    for code in complete.index.astype(str):
        if code == target:
            continue
        dist = _weighted_distance(target_z, z_index[code], weights)
        candidates.append((dist, code))

    candidates.sort(key=lambda item: (item[0], item[1]))
    picked = candidates[:limit]

    out: list[SimilarArea] = []
    for dist, code in picked:
        row = complete.loc[code]
        out.append(
            SimilarArea(
                postal_code=code,
                distance=dist,
                price_per_sqm=float(row["price_per_sqm"]),
                pct_change_5y=float(row["pct_change_5y"]),
                pct_change_1y=float(row["pct_change_1y"]),
                rank=row.get("rank", pd.NA),
                percentile=float(row["percentile"])
                if pd.notna(row.get("percentile"))
                else float("nan"),
            )
        )
    return out


def similar_areas_explanation() -> str:
    return SIMILARITY_METHOD_SENTENCE
