## Fix pass result

| Finding | Status | Details |
| --- | --- | --- |
| [blocking] `app/streamlit_app.py:679-683` — Rent vs buy search box crashes with `TypeError` because `search_area_matches()` returns `list[str]` (postal codes) but was indexed as dicts | fixed | d26d62e — now builds labels via a lookup on `prices` by postal code, matching how the Map tab already uses this helper. Verified interactively with `AppTest`: typing "00100" and opening the selectbox no longer raises and shows the correct label. |
| [should fix] `housing_analyzer/rent_vs_buy.py:846-862` — `prepare_gross_rental_yield_dataframe` called `boundary_properties()` per postal code (O(n²) scan over boundaries) | fixed | 2abfd06 — now builds the postal→municipality mapping once via `postal_to_municipality_codes()` (already used elsewhere in the codebase) before the loop. |
| [should fix] `housing_analyzer/rent_vs_buy.py:869-871` — hover text for postal codes with a price but no rent quote showed the full "found" price panel instead of a no-data message | fixed | 2abfd06 — now shows a dedicated "no rent data for this area" hover when price is present but the rent lookup failed, distinct from the true no-price-data hover. |
| [minor] `tests/test_rent_vs_buy.py:1247-1256` — municipality-fallback result in `test_resolve_price_postal_and_municipality_fallback` is computed but never asserted | not fixed | minor, skipped per fix-pass scope |
| [minor] `housing_analyzer/rent_vs_buy.py:487-497`/`~654-657` — redundant `None`/`NaN` checks alongside `pd.isna()` | not fixed | minor cleanup, skipped per fix-pass scope |

Ran `tests/test_rent_vs_buy.py` after each fix (12 passed) and the full suite at the end (247 passed).

<!-- factory:fix-result:all-addressed -->
