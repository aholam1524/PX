"""Price and area metrics."""


def price_per_sqm(price_eur: float, area_sqm: float) -> float:
    """Return price in euros per square metre."""
    if area_sqm <= 0:
        raise ValueError("area_sqm must be positive")
    return price_eur / area_sqm
