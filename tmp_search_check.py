import os

os.environ["HOUSING_USE_FIXTURES"] = "1"

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/streamlit_app.py")
at.run(timeout=90)
assert not at.exception
labels = [getattr(t, "label", None) for t in at.tabs]
rent_idx = labels.index("Rent vs buy")
at.tabs[rent_idx].run(timeout=90)
assert not at.exception
ti = [t for t in at.text_input if t.key == "rent_vs_buy_search"]
assert ti, "search box not found"
ti[0].set_value("00100").run(timeout=90)
print("exception:", at.exception)
assert not at.exception
sb = [s for s in at.selectbox if s.key == "rent_vs_buy_search_pick"]
print("selectbox found:", bool(sb))
if sb:
    print("options:", sb[0].options)
print("OK")
